#!/usr/bin/env bash
set -euo pipefail

HOST="http://10.173.2.72:8091"

while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

export HOST
export MODEL="cursor/dsv31-gb200-tgl-test-02"
export PROMPT="$(cat <<'EOF'
Hi, You are a highly knowledgeable travel expert and city historian with deep familiarity with New York City across culture, history, food, architecture, arts, transportation, and neighborhoods.
Your task is to answer questions about New York City with clarity, depth, and structured reasoning.

Before answering, carefully consider the following background context and constraints. This context is intentionally long and should be treated as important reference material.

New York City is composed of five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. Each borough has distinct cultural, historical, and social characteristics. Manhattan is often associated with finance, Broadway, museums, and iconic landmarks. Brooklyn is known for creative communities, food scenes, and waterfront parks. Queens is one of the most ethnically diverse places in the world. The Bronx is the birthplace of hip-hop and home to important sports and cultural institutions. Staten Island offers suburban characteristics and views of Manhattan Harbor.

Tourism in New York City generally clusters around several categories:
1. Iconic landmarks and skyline experiences
2. Cultural institutions such as museums and theaters
3. Neighborhood exploration and food

Visitors often have limited time and want to maximize impact. Therefore, answers should prioritize experiences that are:
- High cultural or historical significance
- Widely regarded as representative of New York City
- Accessible to first-time visitors

When ranking or selecting activities, consider factors such as:
- Global recognition
- Density of experiences
- Time efficiency
- Seasonal neutrality

Examples of iconic landmarks include (but are not limited to): Central Park, Times Square, the Statue of Liberty, Empire State Building, Brooklyn Bridge, and One World Trade Center.
Examples of cultural institutions include the Metropolitan Museum of Art, Museum of Modern Art, Broadway theaters, and Lincoln Center.
Examples of neighborhood experiences include walking through SoHo, Greenwich Village, Chinatown, Harlem, Williamsburg, and Dumbo.

Food is an important but optional dimension. If included, it should be representative (e.g., New York–style pizza, bagels, deli culture) rather than overly niche.

Tone requirements:
- Clear and confident
- Informative but concise
- Structured with numbered lists when appropriate

Now answer the following question:

What are the top 3 things to do in New York City for a first-time visitor?

In your answer:
- List exactly three items
- For each item, give a short title and a 2–3 sentence explanation
- Focus on experiences rather than specific businesses unless unavoidable
- Assume the reader has limited time but wants a classic New York experience
EOF
)"

# Build JSON safely without jq (handles quotes/newlines correctly)
JSON="$(
python3 - <<'PY'
import json, os
payload = {
  "model": os.environ["MODEL"],
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "stream": True,
  "stream_options": {"include_usage": True},
  "temperature": 0.0,
}
print(json.dumps(payload))
PY
)"

curl -sS -N -X POST "$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$JSON" | while IFS= read -r line; do
  if [[ "$line" == data:* ]]; then
    data="${line#data: }"
    if [[ "$data" != "[DONE]" && -n "$data" ]]; then
      printf '%s' "$data" | python3 -c "
import sys, json
try:
    chunk = json.load(sys.stdin)
    content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
    if content:
        print(content, end='', flush=True)
except: pass
"
    fi
  fi
done
echo
