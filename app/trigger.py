"""CronJob entrypoint: start what is due, or one flow, then let the Istio sidecar exit. No business logic lives here.

  python -m app.trigger dispatch                          the one CronJob (every minute): POST /schedules/dispatch
  python -m app.trigger gsc_sync_flow                     one flow now (kept for a manual Job or an old CronJob)
  python -m app.trigger report_flow '{"period": "monthly"}'

`dispatch` asks the running seo-agent to start every schedule in the database that is due (app/schedules.py; the
schedules are edited in the CrewAI console). A flow name calls `POST /flows/{name}/kickoff` with the service token and
`triggered_by=cron`; a 200 "existing" answer (the previous run is still going) counts as success. Flows run in the
service (one place holds the model, the locks and the run records). In the mesh a Job's sidecar would keep the pod
alive after the container exits, so the trigger calls /quitquitquit at the end (harmless without a sidecar).
"""

from __future__ import annotations

import json
import os
import re
import sys

import httpx

# A flow name: lower-case letters, digits and underscores, starting with a letter (ga4_sync_flow has a digit).
NAME = re.compile(r"^[a-z][a-z0-9_]{2,39}_flow$")


def main(argv: list[str]) -> int:
    dispatch = bool(argv) and argv[0] == "dispatch" and len(argv) == 1
    if not argv or not (dispatch or NAME.match(argv[0])):
        print(json.dumps({"event": "trigger", "error": "usage: python -m app.trigger dispatch | <flow_name> [json inputs]"}))
        return 2
    name = argv[0]
    try:
        inputs = json.loads(argv[1]) if len(argv) > 1 else {}
    except ValueError:
        print(json.dumps({"event": "trigger", "flow": name, "error": "inputs are not JSON"}))
        return 2
    url = os.environ.get("SEO_AGENT_URL", "http://seo-agent.jithendar:8080").rstrip("/")
    token = os.environ.get("SEO_AGENT_TOKEN", "")
    code = 0
    try:
        headers = {"Authorization": f"Bearer {token}"}
        if dispatch:
            r = httpx.post(f"{url}/schedules/dispatch", headers=headers, timeout=60)
        else:
            r = httpx.post(f"{url}/flows/{name}/kickoff", json={"inputs": inputs, "triggered_by": "cron"}, headers=headers, timeout=60)
        body = r.json() if r.content else None
        print(json.dumps({"event": "trigger", "flow": name, "status": r.status_code, "result": body}))
        code = 0 if r.status_code in (200, 202) else 1
    except (httpx.HTTPError, ValueError) as e:
        print(json.dumps({"event": "trigger", "flow": name, "error": type(e).__name__}))
        code = 1
    finally:
        try:
            httpx.post("http://127.0.0.1:15020/quitquitquit", timeout=5)
        except httpx.HTTPError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
