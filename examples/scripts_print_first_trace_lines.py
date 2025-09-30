import os, json, asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pk = os.environ['LANGFUSE_PUBLIC_KEY']
sk = os.environ['LANGFUSE_SECRET_KEY']
host = os.environ.get('LANGFUSE_HOST','https://cloud.langfuse.com')

async def parse_envelope(res):
    if not hasattr(res, 'content') or not res.content or not res.content[0].text:
        return None, None
    raw = res.content[0].text
    try:
        payload = json.loads(raw)
    except Exception:
        # Not JSON; return raw
        return raw, None
    if isinstance(payload, dict) and 'data' in payload:
        return payload.get('data'), payload.get('metadata')
    return payload, None

async def main():
    params = StdioServerParameters(command='uv', args=['run','-m','langfuse_mcp','--public-key',pk,'--secret-key',sk,'--host',host,'--log-level','INFO','--log-to-console'])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Step 1: get one trace ID via compact mode
            res = await session.call_tool('fetch_traces', {'age': 10080, 'limit': 1, 'page': 1, 'output_mode': 'compact'})
            data, meta = await parse_envelope(res)
            if not isinstance(data, list) or len(data) == 0:
                print('No traces found in the last 7 days.')
                return
            trace_id = data[0].get('id') if isinstance(data[0], dict) else None
            if not trace_id:
                print('ERROR: could not determine trace id from compact response:', data[0])
                return
            # Step 2: fetch full JSON for that trace and print first 20 lines
            res2 = await session.call_tool('fetch_trace', {'trace_id': trace_id, 'include_observations': True, 'output_mode': 'full_json_string'})
            if not hasattr(res2, 'content') or not res2.content or not res2.content[0].text:
                print('ERROR: empty trace detail response')
                return
            try:
                detail_payload = json.loads(res2.content[0].text)
            except Exception as e:
                print('ERROR: unable to parse trace JSON:', e)
                print('RAW:', res2.content[0].text[:200])
                return
            # Handle envelope for full_json_string: server returns a JSON string (not envelope)
            detail = detail_payload if isinstance(detail_payload, dict) else detail_payload
            pretty = json.dumps(detail, indent=2, default=str)
            lines = pretty.splitlines()
            for i, line in enumerate(lines[:20], 1):
                print(line)
            if len(lines) > 20:
                print(f"\n... (truncated, total lines: {len(lines)})")

asyncio.run(main())
