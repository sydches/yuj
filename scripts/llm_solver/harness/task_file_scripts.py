"""Static task-file programs executed only through the selected namespace."""

_DISCOVER = r'''
for utility in "$@"; do
    location=$(type -P -- "$utility") || location=''
    printf '%s\0%s\0' "$utility" "$location"
done
'''

_DIGEST_BATCH = r'''
digest_batch() {
    local readlink_bin=$1 input_fd code target index record descriptor
    local -a batch targets inputs opened
    local -A labels
    while :; do
        mapfile -d '' -n 64 -t batch
        (( ${#batch[@]} )) || break
        mapfile -d '' -t targets < <("$realpath_bin" -m -z -- "${batch[@]}")
        wait "$!" || return 74
        (( ${#targets[@]} == ${#batch[@]} )) || return 74
        inputs=(); labels=()
        for index in "${!targets[@]}"; do
            target=${targets[$index]}
            case "$target" in "$root"|"${root%/}/"*) ;; *) return 77 ;; esac
            [[ -e "$target" ]] || return 66
            [[ -f "$target" ]] || return 74
            exec {input_fd}< "$target" || return 74
            inputs+=("/proc/self/fd/$input_fd")
            labels[$input_fd]=${batch[$index]}
        done
        mapfile -d '' -t opened < <("$readlink_bin" -z -- "${inputs[@]}")
        wait "$!" || return 74
        (( ${#opened[@]} == ${#inputs[@]} )) || return 74
        for target in "${opened[@]}"; do
            case "$target" in "$root"|"${root%/}/"*) ;; *) return 77 ;; esac
        done
        "$utility" -z -- "${inputs[@]}" |
            while IFS= read -r -d '' record; do
                descriptor=${record##*/}
                [[ -v labels[$descriptor] ]] || exit 74
                printf '%s\0%s\0' "${labels[$descriptor]}" "${record%% *}"
            done
        code=(${PIPESTATUS[@]})
        for input_fd in "${!labels[@]}"; do exec {input_fd}<&-; done
        (( code[0] == 0 && code[1] == 0 )) || return 74
    done
}
'''

_RESOLVE_FILES = r'''
resolve_files() {
    local index target partial=${1:-}
    local -a batch resolved
    while :; do
        mapfile -d '' -n 64 -t batch
        (( ${#batch[@]} )) || break
        mapfile -d '' -t resolved < <("$realpath_bin" -m -z -- "${batch[@]}")
        wait "$!" || return 74
        (( ${#resolved[@]} == ${#batch[@]} )) || return 74
        for index in "${!batch[@]}"; do
            target=${resolved[$index]}
            case "$target" in
                "$root"|"${root%/}/"*) ;;
                *) [[ "$partial" == partial ]] && continue; return 77 ;;
            esac
            if [[ "$partial" != paths && ! -f "$target" ]]; then
                [[ "$partial" == partial ]] && continue
                return 74
            fi
            printf '%s\0%s\0' "${batch[$index]}" "$target"
        done
    done
}
'''

# Arguments and file bytes stay separate from the executable script. Append a
# sentinel when capturing realpath output so filenames ending in LF survive.
_OPERATE = r'''
root=$1; requested=$2; realpath_bin=$3; operation=$4; utility=$5
shift 5
canonical() {
    local result
    result=$("$realpath_bin" -m -- "$2"; code=$?; printf '.'; exit "$code") || return
    result=${result%.}
    result=${result%$'\n'}
    printf -v "$1" '%s' "$result"
}
change_mode() {
    # Some chmod implementations skip a symlink without reporting an error.
    # Do not mistake that skipped change for success or dereference the link.
    [[ ! -L "$3" ]] || { printf 'task symlink changed before chmod' >&2; return 74; }
    local error code mode_fd opened expected
    if ! error=$("$1" --no-dereference "$2" -- "$3" 2>&1); then
        # Older utilities lack this option. A readable entry can instead be
        # changed through a checked open descriptor, never a mutable symlink.
        if [[ $("$1" --help 2>&1) == *--no-dereference* ]]; then
            printf '%s' "$error" >&2; return 74
        fi
        [[ ! -L "$3" ]] || return 74
        canonical expected "$3" || return 74
        case "$expected" in "$root"|"${root%/}/"*) ;; *) return 77 ;; esac
        exec {mode_fd}< "$3" || return 74
        opened=$("$readlink_bin" -- "/proc/self/fd/$mode_fd"; code=$?; printf '.'; exit "$code") || return 74
        opened=${opened%.}; opened=${opened%$'\n'}
        [[ "$opened" == "$expected" ]] || return 74
        "$1" "$2" -- "/proc/self/fd/$mode_fd"
        code=$?
        exec {mode_fd}<&-
        (( code == 0 )) || return "$code"
    fi
    [[ ! -L "$3" ]] || { printf 'task symlink changed during chmod' >&2; return 74; }
}
enter_directory() {
    local opened_parent
    cd -P -- "$1" || return 74
    opened_parent=$("$2" -- /proc/self/cwd; code=$?; printf '.'; exit "$code") || return 74
    opened_parent=${opened_parent%.}; opened_parent=${opened_parent%$'\n'}
    case "$opened_parent" in
        "$root"|"${root%/}/"*) ;;
        *) printf 'opened parent escapes task root' >&2; return 77 ;;
    esac
}
canonical root "$root" || exit 74
entry="$root/$requested"
case "$operation:${entry##*/}" in
    *:.|*:..|*:"") canonical target "$entry" || exit 74 ;;
    symlink:*|readlink:*|lstat:*|read_entry:*|unlink:*|replace:*|link:*|create:*|rmdir:*)
        canonical parent "${entry%/*}" || exit 74
        target="$parent/${entry##*/}"
        ;;
    *) canonical target "$entry" || exit 74 ;;
esac
case "$target" in
    "$root"|"${root%/}/"*) ;;
    *) printf 'path escapes task root' >&2; exit 77 ;;
esac
# Establish absent paths with native predicates, not translated stderr. Search
# permission on the nearest existing ancestor is required to claim absence.
ancestor=$target
while [[ ! -e "$ancestor" && ! -L "$ancestor" && "$ancestor" != / ]]; do
    ancestor=${ancestor%/*}
    [[ -n "$ancestor" ]] || ancestor=/
done
if [[ -d "$ancestor" && ! -x "$ancestor" &&
      ( "$ancestor" != "$target" ||
        ( "$operation" != chmod && "$operation" != stat && "$operation" != lstat && "$operation" != mkdir ) ) ]]; then
    printf 'directory search denied' >&2; exit 77
fi
# These operations use one checked directory and relative names. Metadata
# and entry operations do not follow the final component at execution.
if [[ "$operation" == unlink && "${2:-}" == -f && ! -e "$target" && ! -L "$target" ]]; then
    exit 0
fi
if [[ "$operation" == unlink_entries && ! -e "$target" && ! -L "$target" ]]; then
    exit 0
fi
case "$operation" in
    stat|lstat|list|scandir|entry_modes|entry_identities|readlink|read_entry) [[ -e "$target" || -L "$target" ]] || exit 66 ;;
    symlink) [[ -e "$target" || -L "$target" ]] || { printf false; exit 0; } ;;
esac
case "$operation" in
    write|create|replace|link|unlink|unlink_entries|rmdir|stat|lstat|list|scandir|entry_modes|entry_identities|readlink|read_entry|symlink|chmod|search_files)
        readlink_bin=$1; shift
        if [[ "$operation" == list || "$operation" == scandir || "$operation" == entry_modes || "$operation" == entry_identities || "$operation" == unlink_entries || "$target" == "$root" ||
              ( "$operation" == search_files && -d "$target" ) ]]; then
            directory=$target
            target=.
        else
            directory=${target%/*}
            [[ -n "$directory" ]] || directory=/
            target="./${target##*/}"
        fi
        enter_directory "$directory" "$readlink_bin" || exit
        ;;
esac
case "$operation" in
    resolve) printf '%s\0' "$target" ;;
    glob) native_glob "$@" ;;
    search_batch) native_search "$@" ;;
    digest_batch) digest_batch "$@" ;;
    resolve_files) resolve_files "$@" ;;
    search_files)
        # The checked operation already knows which scope supplied these names.
        if [[ "$target" == . ]]; then printf 'd\0'; else printf 'f\0'; fi
        exec "$utility" --files --null --no-follow "$@" -- "$target"
        ;;
    symlink)
        if [[ -L "$target" ]]; then printf true; else printf false; fi
        ;;
    entry_identities)
        for name in "$@"; do
            [[ -n "$name" && "$name" != */* && "$name" != . && "$name" != .. ]] || exit 77
            if [[ -e "./$name" || -L "./$name" ]]; then
                identity=$("$utility" -c '%d %i %f' -- "./$name") || exit 74
                printf '%s\0%s\0' "$name" "$identity"
            fi
        done
        ;;
    entry_modes)
        entries=()
        for name in "$@"; do
            [[ "$name" != */* && "$name" != . && "$name" != .. ]] || exit 77
            if [[ -e "./$name" || -L "./$name" ]]; then entries+=("$name"); fi
        done
        ((${#entries[@]})) || exit 0
        exec "$utility" --printf='%n\0%f\0' -- "${entries[@]}"
        ;;
    read_entry)
        if [[ -L "$target" ]]; then
            printf 'a000\0'
            exec "$readlink_bin" -z -- "$target"
        fi
        [[ -f "$target" ]] || { printf 'unsupported task entry type' >&2; exit 74; }
        exec {input_fd}< "$target" || exit 74
        opened_path=$("$readlink_bin" -- "/proc/self/fd/$input_fd"; code=$?; printf '.'; exit "$code") || exit 74
        opened_path=${opened_path%.}; opened_path=${opened_path%$'\n'}
        case "$opened_path" in
            "$root"|"${root%/}/"*) ;;
            *) printf 'opened file escapes task root' >&2; exit 77 ;;
        esac
        [[ -f "/proc/self/fd/$input_fd" ]] || exit 74
        "$1" -L --printf='%f\0' -- "/proc/self/fd/$input_fd" || exit 74
        exec "$utility" -- <&"$input_fd"
        ;;
    read|read_observation|read_range|search)
        [[ -e "$target" ]] || exit 66
        [[ ! -d "$target" ]] || exit 73
        readlink_bin=$1; shift
        # Resolve the opened file's kernel path, not the requested name again.
        # A parent link can change between realpath and open. No file bytes
        # enter the response until this descriptor passes containment.
        exec {input_fd}< "$target" || exit 74
        opened_path=$("$readlink_bin" -- "/proc/self/fd/$input_fd"; code=$?; printf '.'; exit "$code") || exit 74
        opened_path=${opened_path%.}; opened_path=${opened_path%$'\n'}
        case "$opened_path" in
            "$root"|"${root%/}/"*) ;;
            *) printf 'opened file escapes task root' >&2; exit 77 ;;
        esac
        [[ ! -d "/proc/self/fd/$input_fd" ]] || exit 73
        if [[ "$operation" == read_observation ]]; then
            [[ -f "/proc/self/fd/$input_fd" ]] || exit 74
            export LC_ALL=C
            printf '%s\0' "$opened_path"
            "$1" -L --printf='%f\0%s\0%Y\0%Z\0%i\0%d\0%y\0%z\0' -- "/proc/self/fd/$input_fd" || exit 74
            "$utility" -- <&"$input_fd" || exit 74
            printf '\0'
            "$1" -L --printf='%f\0%s\0%Y\0%Z\0%i\0%d\0%y\0%z\0' -- "/proc/self/fd/$input_fd" || exit 74
            exit 0
        fi
        if [[ "$operation" == search ]]; then
            exec "$utility" -n --no-filename --color=never -- "$1" - <&"$input_fd"
        fi
        if [[ "$operation" == read_range ]]; then
            exec "$utility" iflag=skip_bytes,count_bytes skip="$1" count="$2" status=none <&"$input_fd"
        fi
        exec "$utility" -- <&"$input_fd"
        ;;
    write)
        # Canonical resolution permits existing contained aliases. At the
        # actual open, refuse a new final symlink instead of following it.
        # O_WRONLY preserves write-only files; nocreat/excl avoid accepting a
        # different creation state between this predicate and the open.
        conversion=excl
        [[ ! -e "$target" && ! -L "$target" ]] || conversion=nocreat
        exec "$utility" of="$target" oflag=nofollow conv="$conversion" status=none
        ;;
    create)
        mktemp_bin=$1; ln_bin=$2; rm_bin=$3
        [[ ! -e "$target" && ! -L "$target" ]] || exit 75
        temporary=$("$mktemp_bin" --tmpdir="${target%/*}" .yuj-output-XXXXXXXXXX) || exit
        trap '"$rm_bin" -f -- "$temporary"' EXIT
        "$utility" of="$temporary" oflag=nofollow conv=nocreat status=none || exit
        if "$ln_bin" -T -- "$temporary" "$target"; then
            exit 0
        fi
        [[ ! -e "$target" && ! -L "$target" ]] || exit 75
        exit 74
        ;;
    replace)
        mktemp_bin=$1; mv_bin=$2; rm_bin=$3; chmod_bin=$4; rmdir_bin=$5; mode=$6
        temporary=$("$mktemp_bin" --tmpdir="${target%/*}" .yuj-restore-XXXXXXXXXX) || exit
        trap '"$rm_bin" -f -- "$temporary"' EXIT
        # Flush the writer's open file before applying its final permissions.
        # A later path-based sync can be redirected or denied by that mode.
        "$utility" of="$temporary" oflag=nofollow conv=nocreat,fdatasync status=none || exit
        change_mode "$chmod_bin" "$mode" "$temporary" || exit
        if [[ -d "$target" && ! -L "$target" ]]; then
            "$rmdir_bin" -- "$target" || exit
        fi
        "$mv_bin" -T -- "$temporary" "$target"
        ;;
    link) exec "$utility" -s -T -- "$1" "$target" ;;
    stat|lstat)
        [[ -e "$target" || -L "$target" ]] || exit 66
        export LC_ALL=C
        exec "$utility" --printf='%f\0%s\0%Y\0%Z\0%i\0%d\0%y\0%z\0' -- "$target"
        ;;
    readlink) exec "$utility" -z -- "$target" ;;
    mkdir)
        readlink_bin=$1; shift
        if [[ "$target" == "$root" ]]; then
            enter_directory "$root" "$readlink_bin" || exit
            exec "$utility" "$@" -- .
        fi
        directory=${target%/*}
        [[ -n "$directory" ]] || directory=/
        if [[ "${1:-}" == -p ]]; then
            while [[ ! -e "$directory" && ! -L "$directory" && "$directory" != / ]]; do
                directory=${directory%/*}
                [[ -n "$directory" ]] || directory=/
            done
        fi
        remaining=${target#"$directory"}
        remaining=${remaining#/}
        enter_directory "$directory" "$readlink_bin" || exit
        if [[ "${1:-}" != -p ]]; then
            exec "$utility" -- "./$remaining"
        fi
        while [[ "$remaining" == */* ]]; do
            component=${remaining%%/*}
            # Native mkdir -p grants owner write/search on new intermediates.
            # Preserve every other umask bit, and leave the final directory's
            # umask unchanged. Existing directory modes are not modified.
            (umask u+wx; "$utility" -p -- "./$component") || exit
            enter_directory "./$component" "$readlink_bin" || exit
            remaining=${remaining#*/}
        done
        "$utility" -p -- "./$remaining" || exit
        [[ ! -L "./$remaining" ]] || { printf 'task symlink changed during mkdir' >&2; exit 74; }
        ;;
    chmod)
        change_mode "$utility" "$1" "$target"
        ;;
    rmdir) exec "$utility" -- "$target" ;;
    unlink) exec "$utility" "$@" -- "$target" ;;
    unlink_entries) exec "$utility" -f -- "$@" ;;
    list) exec "$utility" "$target" -mindepth 1 -maxdepth 1 -print0 ;;
    scandir) exec "$utility" "$target" -mindepth 1 -maxdepth 1 -printf '%f\0%y\0' ;;
    *) printf 'unsupported task file operation' >&2; exit 64 ;;
esac
'''
