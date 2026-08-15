#!/bin/bash
# Step 3: does a job actually receive a /sptgrid SciToken, and can it move real
# data both ways against the SPT dCache?
#
# This is the one link in the staging design that configuration inspection
# cannot settle. The AP declares
#   LOCAL_CREDMON_AUTHZ_TEMPLATE = read:/sptgrid write:/sptgrid
# and an AddSciToken job transform, and curl_plugin is built to read
# $_CONDOR_CREDS and send a bearer header. Whether all of that composes into a
# working transfer has never been observed.
#
# READ is tested by HTCondor itself: transfer_input_files carries a davs:// URL,
# so if this script runs at all, the pull already succeeded. We verify the bytes.
# WRITE is tested here, by hand, with the same token.

say() { echo "### $1"; }

EXPECT_MD5=15609acd1c5d4003a6544a4618dcf99e
EXPECT_SIZE=36378653
DCACHE=https://osg-gridftp.grid.uchicago.edu:2880

say SITE
echo "HOST=$(hostname)"
echo "SITE=${GLIDEIN_ResourceName:-unset}"

say TOKEN
# The credential HTCondor hands the job. If this is empty the AddSciToken
# transform did not fire and nothing below can work.
echo "_CONDOR_CREDS=${_CONDOR_CREDS:-unset}"
ls -la "${_CONDOR_CREDS:-/nonexistent}" 2>&1 | head -5
TOK=""
for cand in "$_CONDOR_CREDS"/scitokens.use "$_CONDOR_CREDS"/*.use; do
    [ -f "$cand" ] && { TOK="$cand"; break; }
done
if [ -n "$TOK" ]; then
    echo "token_file=$TOK ($(stat -c%s "$TOK") bytes)"
    # Decode the payload claims without printing the signature.
    python3 - "$TOK" <<'PY' 2>/dev/null || echo "(claim decode unavailable)"
import base64, json, sys
raw = open(sys.argv[1]).read().strip()
p = raw.split('.')[1]
p += '=' * (-len(p) % 4)
c = json.loads(base64.urlsafe_b64decode(p))
for k in ("iss", "aud", "scope", "exp", "sub"):
    if k in c:
        print(f"  claim {k}: {c[k]}")
PY
else
    echo "NO TOKEN FOUND -- read transfer must have used something else"
fi

say READ_VERIFY
# HTCondor already pulled this via transfer_input_files = davs://...
if [ -f spt3g.tgz ]; then
    got_size=$(stat -c%s spt3g.tgz)
    got_md5=$(md5sum spt3g.tgz | cut -d' ' -f1)
    echo "size=$got_size (expect $EXPECT_SIZE)"
    echo "md5 =$got_md5 (expect $EXPECT_MD5)"
    if [ "$got_size" = "$EXPECT_SIZE" ] && [ "$got_md5" = "$EXPECT_MD5" ]; then
        echo "READ_OK yes -- 36 MB pulled from dCache by URL, bytes match"
    else
        echo "READ_OK NO -- file arrived corrupt"
    fi
else
    echo "READ_OK NO -- spt3g.tgz absent from the sandbox"
fi

say WRITE_TEST
# Push a small file back to the user's own dCache area with the job's token.
if [ -n "$TOK" ]; then
    OUT=/sptgrid/user/cailmdaley/osg_test/probe4_$(hostname)_$$.txt
    echo "written by probe4 from $(hostname) at $(date -u +%FT%TZ)" > payload.txt
    code=$(curl -sS -o /tmp/put.out -w '%{http_code}' -T payload.txt \
             -H "Authorization: Bearer $(cat "$TOK")" \
             "$DCACHE$OUT" 2>/tmp/put.err)
    echo "PUT $OUT -> HTTP $code"
    head -3 /tmp/put.err 2>/dev/null | sed 's/^/  err: /'
    case "$code" in
        20*|30*) echo "WRITE_OK yes" ;;
        *)       echo "WRITE_OK NO" ;;
    esac
else
    echo "skipped -- no token"
fi

say DONE
