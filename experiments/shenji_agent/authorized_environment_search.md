# Authorized Environment Search

Date: 2026-04-29

## Target

- SDK `.engine_config` bound MAC: `02:42:ac:11:00:45`
- Current non-loopback MAC: `02:42:ac:11:00:02`

## Current Interfaces

Generated evidence:

- `experiments/shenji_agent/current_interfaces.txt`
- `experiments/shenji_agent/current_mac_addresses.txt`

Observed interfaces:

- `lo`: `00:00:00:00:00:00`
- `eth0@if192`: `02:42:ac:11:00:02`

No current interface has MAC `02:42:ac:11:00:45`.

## Container/Namespace Search

- Docker CLI: not installed (`docker: command not found`).
- Docker Compose: not installed (`docker compose` and `docker-compose` unavailable).
- Docker socket: `/var/run/docker.sock` absent.
- Podman: not installed (`podman: command not found`).
- Buildah/Nerdctl/Singularity/Apptainer: not installed.
- `/var/run/netns`: absent/empty.
- `/proc/*/ns/net`: 7 visible processes, 1 unique network namespace (`net:[4026536007]`).
- Current namespace MAC list contains only `02:42:ac:11:00:02` and loopback.

No existing accessible container or namespace with MAC `02:42:ac:11:00:45` was found.

## Permission Matrix

- User identity: `uid=0(root) gid=0(root) groups=0(root)` inside the container.
- Effective capabilities: `cap_chown`, `cap_dac_override`, `cap_fowner`, `cap_fsetid`, `cap_kill`, `cap_setgid`, `cap_setuid`, `cap_setpcap`, `cap_net_bind_service`, `cap_net_raw`, `cap_sys_chroot`, `cap_mknod`, `cap_audit_write`, `cap_setfcap`.
- `CAP_NET_ADMIN`: absent.
- `CAP_SYS_ADMIN`: absent.
- User namespace creation: `unshare --user --map-root-user true` fails with `Operation not permitted`.
- Slurm/qsub/srun: not installed.
- Kubernetes CLI: not installed.

Current permissions allow repository diagnostics, tests, SDK preflight, and current-environment smoke. They do not allow creating a new network namespace, assigning a container MAC, or reconfiguring the current interface. MAC spoofing was not attempted.

## Remote Runner Search

Project scripts found:

- `scripts/bootstrap_remote.py`
- `scripts/sync_remote_artifacts.py`
- `scripts/launch_detached_compare.py`

Remote default discovered in scripts:

- `lizilong@146.56.220.99`
- port `21427`
- repo path `/remote-home/lizilong/shenji_agent`

Attempted non-secret preflight over SSH:

```bash
ssh -p 21427 -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
  lizilong@146.56.220.99 \
  'hostname; cd /remote-home/lizilong/shenji_agent && python3.10 scripts/check_sdk_auth_environment.py --output experiments/shenji_agent/sdk_auth_environment_check_remote_probe.json'
```

Result:

- SSH executable exists at `/usr/bin/ssh`.
- Local `~/.ssh` has `authorized_keys` and `known_hosts`; no private key file was present.
- Connection failed with `Permission denied (publickey,password)`.
- No secrets were printed or uploaded.

The remote runner path is present but not accessible from this current environment.

## Direct SDK/Offline Path

- Direct Python path resolves real backend `train_env.TrainEnv`.
- Local docs describe `run_offline_games.py` as offline/local development, but it imports the same `TrainEnv`.
- No official local/offline auth bypass or alternate license path was found.
- Current direct execution remains blocked by `failed_mac_mismatch`.

## Result

No accessible authorized environment was found under current permissions. The only remaining legal continuation path is to run the repository in an authorized runtime whose non-loopback MAC is `02:42:ac:11:00:45`, or obtain an official regenerated `.engine_config` for the current runtime MAC.
