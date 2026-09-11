"""Observe the Docker client's selected context and responding engine."""
from dataclasses import dataclass
import hashlib
import json
import subprocess

from .time_budget import command_time_budget, execution_deadline, remaining_before


@dataclass(frozen=True)
class DockerIdentity:
    engine_id: str
    context_fingerprint: str


def observe_docker_identity(*, timeout=None):
    """Use backend responses, keeping context contents out of trace records.

    This observation detects a changed target before execution. It does not
    pin a network connection or establish that a daemon shares the host kernel.
    """
    def query(arguments):
        result = subprocess.run(['docker', *arguments], capture_output=True,
                                text=True, check=True,
                                timeout=remaining_before(execution_deadline()))
        return json.loads(result.stdout)

    with command_time_budget(0 if timeout is None else timeout):
        contexts = query(['context', 'inspect'])
        if not isinstance(contexts, list) or len(contexts) != 1 or not isinstance(contexts[0], dict):
            raise ValueError('Docker did not return one selected context')
        context = contexts[0]
        if not isinstance(context.get('Name'), str) or not context['Name']:
            raise ValueError('Docker context has no identity')
        endpoints = context.get('Endpoints')
        if not isinstance(endpoints, dict) or not isinstance(endpoints.get('docker'), dict):
            raise ValueError('Docker context has no engine endpoint')
        host = endpoints['docker'].get('Host')
        if not isinstance(host, str) or not host:
            raise ValueError('Docker context has no engine endpoint')
        engine_id = query(['info', '--format', '{{json .ID}}'])
        if not isinstance(engine_id, str) or not engine_id or engine_id.strip() != engine_id:
            raise ValueError('Docker engine has no observed identity')
        fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return DockerIdentity(engine_id, fingerprint)
