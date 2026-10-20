#!/usr/bin/env bash
# Experiment starter file shared by every wrapper bash scripts

run_role() {
  local target="${1:-both}"
  [[ $# -gt 0 ]] && shift
  case "$target" in
    control|misaligned)
      launch_role "${target}_model" ${ROLE_OVERRIDES[@]+"${ROLE_OVERRIDES[@]}"} "$@"
      ;;
    both)
      declare -F precheck_both >/dev/null && precheck_both
      local log_dir="${LOG_DIR:-outputs/$SHORT_NAME/$BASE_MODEL/$MAX_TOKENS/seed_$SEED/logs}"
      mkdir -p "$log_dir"
      local wrapper="$0" role roles=(control misaligned) pids=()
      for role in "${roles[@]}"; do
        (
          declare -F env_for_role >/dev/null && env_for_role "$role"
          exec bash "$wrapper" "$role" "$@"
        ) > "$log_dir/$role.log" 2>&1 &
        pids+=($!)
        echo "[both] $role pid=$!"
      done
      echo "[both] logs: $log_dir"
      # Wait on each pid individually: `wait "${pids[@]}"` returns only the last pid's status.
      local i fail=0
      for i in "${!roles[@]}"; do
        if ! wait "${pids[$i]}"; then
          echo "[both] ${roles[$i]} model FAILED (see $log_dir/${roles[$i]}.log)" >&2
          fail=1
        fi
      done
      return "$fail"
      ;;
    *)
      echo "usage: $(basename "$0") [both|control|misaligned] [hydra overrides...]" >&2
      return 1
      ;;
  esac
}
