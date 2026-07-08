#!/usr/bin/env bash
# Pull results from all launched wide-sweep pods + print each model's referent sign.
cd /Users/jeb/experimentation || exit 1
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
K=$HOME/.ssh/id_ed25519_runpod
OUT=results/cross_arch_wide
mkdir -p "$OUT"
DONE=0; RUNNING=0
for lg in "$S"/launch_w*.log; do
  ep=$(grep -aoE "LAUNCHED w[0-9]+ at [0-9.]+:[0-9]+" "$lg" 2>/dev/null | tail -1 | awk '{print $4}')
  [ -z "$ep" ] && continue
  ip=${ep%:*}; port=${ep#*:}
  # pull any result json (skip _ prefixed)
  tmp=$(ssh -i $K -p $port -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 root@$ip \
    'cd /workspace/exp 2>/dev/null && for f in results/cross_arch/*.json; do [ -f "$f" ] && [ "$(basename $f | cut -c1)" != "_" ] && cat "$f"; done' 2>/dev/null)
  if [ -n "$tmp" ]; then
    echo "$tmp" | python3 -c "
import json,sys
try:
  d=json.load(sys.stdin)
except: sys.exit(0)
m=d.get('model') or d.get('model_id') or '?'
slug=m.replace('/','__')
open('$OUT/'+slug+'.json','w').write(json.dumps(d))
st=d.get('status'); r=(d.get('by_category_robust') or {}).get('referent',{}) or {}
ci=r.get('raw_EB_ci'); sign='?' 
if ci and None not in ci:
  mid=sum(ci)/2; sign='POS' if ci[0]>0 else ('NEG' if ci[1]<0 else 'null(spans0)'); 
print(f'  {slug}: status={st} referent_ci={ci} sign={sign}')
" 2>/dev/null && DONE=$((DONE+1))
  else
    RUNNING=$((RUNNING+1))
  fi
done
echo "HARVEST: $DONE with-result, $RUNNING no-result-yet"
