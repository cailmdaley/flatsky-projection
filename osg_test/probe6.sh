#!/bin/bash
# Step 3c: WHY does dCache 401 a validly-signed AP token?
#
# probe5 established: TLS trust works (401 not cert error), the token arrives
# with sub=cailmdaley scope="read:/sptgrid write:/sptgrid" and the doors in aud,
# and the issuer's openid-configuration + JWKS (kid f29b) resolve publicly.
# So the signature is verifiable and the rejection is server-side policy:
# issuer not trusted by gPlazma, a scope-prefix convention mismatch, or a
# mapping failure. This probe captures the evidence: full claim set, verbose
# response headers and bodies, a path-prefix variant, and a macaroon request.
#
# dCache throttles repeated failed logins with 429s, so attempts are few and
# spaced 30 s apart, most diagnostic first.

say() { echo "### $1"; }

DCACHE=https://osg-gridftp.grid.uchicago.edu:2880
SRC=/sptgrid/user/cailmdaley/tarballs/spt3g.tgz
CADIR=/cvmfs/spt.opensciencegrid.org/data/CA/certificates

say SITE
echo "HOST=$(hostname)  SITE=${GLIDEIN_ResourceName:-unset}  DATE=$(date -u +%FT%TZ)"

say TOKEN_FULL
TOK=""
for cand in "$_CONDOR_CREDS"/scitokens.use "$_CONDOR_CREDS"/*.use; do
    [ -f "$cand" ] && { TOK="$cand"; break; }
done
if [ -z "$TOK" ]; then echo "NO TOKEN"; exit 0; fi
/usr/bin/python3 - "$TOK" <<'PY'
import base64, json, sys, time
raw = open(sys.argv[1]).read().strip()
for i, part in enumerate(raw.split('.')[:2]):
    part += '=' * (-len(part) % 4)
    d = json.loads(base64.urlsafe_b64decode(part))
    print(("header:" if i == 0 else "payload:"), json.dumps(d, sort_keys=True))
    if i == 1:
        now = time.time()
        for k in ("iat", "nbf", "exp"):
            if k in d: print(f"  {k} - now = {d[k]-now:+.0f}s")
PY

attempt() {  # attempt <label> <curl args...>
    say "$1"; shift
    curl -sS --capath "$CADIR" -H "Authorization: Bearer $(cat "$TOK")" \
         -D /tmp/hdr -o /tmp/body -w 'HTTP %{http_code}\n' "$@" 2>&1
    grep -iE '^(www-authenticate|content-type|HTTP/)' /tmp/hdr | sed 's/^/  hdr: /'
    head -c 500 /tmp/body | sed 's/^/  body: /'
    echo
}

# 1. The canonical read, verbose. The body/WWW-Authenticate is the evidence.
attempt GET_CANONICAL "$DCACHE$SRC" -X GET -r 0-1023
sleep 30

# 2. Path without the /sptgrid prefix: tests the gPlazma scope-prefix convention
#    (if the door root already IS /sptgrid, the canonical path double-counts it).
attempt GET_NO_PREFIX "$DCACHE/user/cailmdaley/tarballs/spt3g.tgz" -X GET -r 0-1023
sleep 30

# 3. Macaroon request: a different auth code path in dCache; the error message
#    (if any) is usually more articulate than the WebDAV door's.
attempt MACAROON "$DCACHE$SRC" -X POST \
    -H 'Content-Type: application/macaroon-request' \
    -d '{"caveats": ["activity:DOWNLOAD"], "validity": "PT1H"}'

say DONE
