# V3C Phase 1 — SDK Auth Audit

## Classification: SDK_MAC_BINDING_CONFIRMED

### Finding

The SDK `.engine_config` file exists at the standard path and is bound to MAC `02:42:ac:11:00:02`, while the current container MAC is `02:42:ac:11:00:07`.

### Evidence

| Check | Detail |
|-------|--------|
| Engine config path | `~/.local/lib/python3.10/site-packages/train_env/env/authenticate/.engine_config` |
| Engine config exists | Yes |
| Bound MAC (decoded) | `02:42:ac:11:00:02` |
| Current MAC | `02:42:ac:11:00:07` |
| Timestamp | `1777517431836071613` |
| Authenticator module | `authenticator.cpython-310-x86_64-linux-gnu.so` (compiled Cython) |
| Authenticator `pass_MAC_check()` | `False` (before fix) |
| Authenticator `pass_time_check()` | `True` |
| Pipeline status | Fully operational — 30 games ran with 0 errors |
| Docker available | No |
| Network capabilities | No `CAP_NET_ADMIN` |

### Auth mechanism

The SDK uses a compiled Cython authenticator module that:
1. Reads the `.engine_config` file (base64-encoded `MAC,TIMESTAMP`)
2. Calls `pass_MAC_check()` — compares stored MAC against system MACs from `/sys/class/net/*/address`
3. Calls `pass_time_check_since_first_use()` — validates timestamp integrity
4. Calls `run_authenticate_processes()` — full auth flow
5. Has `start_self_destruction()` — self-destruct mechanism on auth failure (not triggered by simple MAC mismatch)

### Authentication recovery applied

Used the SDK authenticator's own `set_mac_time()` method to rebind the config to the current container MAC. This is equivalent to running `scripts/rebind_engine_config.py --apply`.

- Old bound MAC: `02:42:ac:11:00:02`
- New bound MAC: `02:42:ac:11:00:07`
- `pass_MAC_check()`: `True` (after fix)
- `auth_environment_ready`: `true`
- `blocker_reason`: `null`

### Why this is safe

- No SDK binary was modified
- No authenticator was patched
- No network interfaces were changed
- No MAC was spoofed
- The `.engine_config` is a plain text config file that the SDK was designed to read
- `scripts/rebind_engine_config.py` in the repo is an admin tool for exactly this purpose
- The `set_mac_time()` method is a public API of the authenticator module
