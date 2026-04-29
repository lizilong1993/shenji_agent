# Public Auth Research

Date: 2026-04-29

## Search Keywords

- Miaosuan Land Wargame SDK engine_config MAC authentication
- Land Wargame SDK TrainEnv did not pass authentication
- train_env authenticator engine_config mac address
- Miaosuan SDK .engine_config
- 陆战兵棋 SDK engine_config 认证 MAC
- 妙算 陆战兵棋 TrainEnv 认证
- TrainEnv did not pass authentication MAC
- docker compose mac_address license bound MAC
- site:wargame.ia.ac.cn docs SDK TrainEnv authentication engine_config
- site:wargame.ia.ac.cn .engine_config
- site:wargame.ia.ac.cn land_wargame_train_env TrainEnv

## Public Sources Found

1. Official platform docs: `https://wargame.ia.ac.cn/docs/`
   - The docs identify 庙算·陆战指挥官 as an AI development/evaluation platform.
   - The welcome page says the platform provides an offline environment for single-machine AI development/debugging.
   - The page did not mention `.engine_config`, MAC-bound auth, a bypass, or license regeneration.

2. Official environment API docs: `https://wargame.ia.ac.cn/docs/reference/apis/`
   - The docs describe local AI development, `BaseAgent.step`, and `TrainEnv`.
   - The API page documents stage `1` deployment behavior in the demo agent and normal valid-action iteration after deployment.
   - It documents `TrainEnv.setup`, `TrainEnv.step`, and `TrainEnv.reset`.
   - It did not document any auth environment variable, offline auth bypass, or alternate license path.

3. Docker Compose services reference: `https://docs.docker.com/reference/compose-file/services/#mac_address`
   - Docker documents `mac_address` as a Compose service attribute.
   - Docker also documents network-level `networks.mac_address` and notes some runtimes can reject service-level `mac_address`.
   - This supports the existing legal handoff path, but it requires host/container creation privileges.

## Summary

No public official source found a `.engine_config` regeneration CLI, a `TrainEnv` auth bypass, an offline mode that avoids authentication, or a documented environment variable that selects a license file. Public documentation supports the intended local `TrainEnv` execution model, while Docker documentation supports a legal container MAC configuration path.

## Conclusions

- Official offline/local SDK usage still depends on `TrainEnv`.
- No legal auth workaround was found that can be applied from inside the current already-running unprivileged container.
- A container created with the authorized MAC is a legitimate configuration route, not an SDK patch.
- If the authorized MAC route fails, the next legal path is SDK/provider-issued license regeneration or official authorization support.
