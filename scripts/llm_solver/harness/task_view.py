"""Observe a task's Linux mount view through its file executor."""
import hashlib
import json
import re

from .task_files import TaskFileError


_VIEW = r'''
root=$1; stat_bin=$2; ignored_count=$3
shift 3
ignored_names=("${@:1:ignored_count}")
shift "$ignored_count"
excluded_paths=("$@")
IFS= read -r boot_id </proc/sys/kernel/random/boot_id || exit 74
observe_entry() {
    local entry_fd entry_owner code=0 key value rest
    exec {entry_fd}< "$1" || return 74
    entry_owner=$BASHPID
    observed_mount=''
    while read -r key value rest; do
        [[ "$key" == mnt_id: ]] && observed_mount=$value
    done <"/proc/self/fdinfo/$entry_fd"
    observed_entry=$("$stat_bin" -L --printf='%d:%i:%f' -- "/proc/$entry_owner/fd/$entry_fd") || code=74
    exec {entry_fd}<&-
    [[ "$observed_mount" =~ ^[0-9]+$ ]] || return 74
    return "$code"
}
observe_entry "$root" || exit 74
root_entry=$observed_entry; root_mount=$observed_mount
printf '%s\0%s\0' "$boot_id" "$root_entry"
cover_record=''
while IFS= read -r line; do
    read -r -a fields <<< "$line"
    (( ${#fields[@]} >= 10 )) || exit 74
    if [[ "${fields[0]}" == "$root_mount" ]]; then
        printf -v cover_record '%s ' "${fields[@]:2}"
        continue
    fi
    printf -v point '%b' "${fields[4]}"
    case "$point" in
        "${root%/}/"*)
            relative=${point#"${root%/}/"}
            [[ "$point" == "$root" ]] && relative=''
            excluded=false
            for name in "${ignored_names[@]}"; do
                case "/$relative/" in */"$name"/*) excluded=true ;; esac
            done
            for prefix in "${excluded_paths[@]}"; do
                case "$relative" in "$prefix"|"$prefix"/*) excluded=true ;; esac
            done
            "$excluded" && continue
            ;;
        *)
            continue
            ;;
    esac
    observe_entry "$point" || exit 74
    [[ "$observed_mount" == "${fields[0]}" ]] || continue
    # Mount and parent IDs are allocated anew for equivalent short-lived
    # namespaces. Keep the observed backing device, root, mount path, flags,
    # propagation fields, filesystem and source instead.
    printf '%s ' "${fields[@]:2}"
    printf '\0%s\0' "$observed_entry"
done </proc/self/mountinfo
[[ -n "$cover_record" ]] || exit 74
printf '%s\0%s\0' "$cover_record" "$root_entry"
'''


def task_view_identity(files, *, root, ignored_dir_names=(), excluded_paths=()):
    """Hash native mount facts; path spellings alone cannot establish a view.

    Mounted entry identities detect replacement bind sources even when their
    path strings and source-file sizes/timestamps remain unchanged. Ignore
    only directory names or root-relative subtrees the consumer never reads.
    This is an observation, not a lock against subsequent mount changes.
    """
    result = files.run(_VIEW, [str(root), files._utility('stat'), str(len(ignored_dir_names)),
                              *sorted(ignored_dir_names), *sorted(excluded_paths)], None)
    fields = result.stdout.split(b'\0')
    if (result.returncode or len(fields) < 5 or len(fields) % 2 != 1
            or fields[-1] or not re.fullmatch(rb'[0-9a-f-]{36}', fields[0])):
        raise TaskFileError('task mount-view observation unavailable')
    if any(not re.fullmatch(rb'\d+:\d+:[0-9a-f]+', item)
           for item in fields[1:-1:2]):
        raise TaskFileError('invalid task mount-entry observation')
    payload = {
        'kernel_boot_id': fields[0].decode('ascii'),
        'root_entry': fields[1].decode('ascii'),
        'mounts': sorted((mount.decode('utf-8', errors='surrogateescape'), entry.decode('ascii'))
                         for mount, entry in zip(fields[2:-1:2], fields[3:-1:2])),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
