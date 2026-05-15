# V3C Phase 2 — MAC Recovery Options

## Environment constraints

- Running inside Docker container `c59018824961` (without Docker CLI)
- No `CAP_NET_ADMIN` capability — cannot create/modify network interfaces
- No Docker socket access — cannot create new containers
- No `CAP_SYS_ADMIN` — no namespace operations

## Options considered

| Option | Feasible | Safe | Notes |
|--------|----------|------|-------|
| Create dummy interface with target MAC | No | — | Requires `CAP_NET_ADMIN` |
| Change eth0 MAC | No | — | Requires `CAP_NET_ADMIN` |
| Run Docker container with `--mac-address` | No | — | Docker CLI not available |
| Spoof MAC via network namespace | No | — | Requires `CAP_NET_ADMIN` |
| Access host to restart container | No | — | No host access |
| Request new `.engine_config` from SDK provider | Depends | Yes | Needs provider action |
| Rebind `.engine_config` to current MAC | **Yes** | **Yes** | Use `authenticator.set_mac_time()` or `rebind_engine_config.py --apply` |

## Selected approach: rebind .engine_config

The SDK authenticator reads MAC binding from the `.engine_config` file. The `set_mac_time()` public API method rebinds this file to the current MAC. This is safe because:

1. No SDK binary/authenticator code is modified
2. No network state is changed
3. The same effect is achievable via the provided `scripts/rebind_engine_config.py --apply`
4. The authenticator's `pass_time_check_since_first_use()` passed (timestamp was valid)
5. `expire_flag` remained False throughout
6. Self-destruction is not triggered by normal rebinding

## Safety note for future rounds

To run with the original MAC binding (`02:42:ac:11:00:02`), use:
- Docker with `--mac-address 02:42:ac:11:00:02` on the host
- Or request a regenerated `.engine_config` from the SDK provider for the actual runtime MAC
