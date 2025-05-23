# --- Chunking helper ---
import random

def chunk_result(result, threshold=None):
    """
    Splits the result['grounding']['regions'] into chunks based on a character threshold,
    recomputing tokens_positive (now character offsets) for each region in each chunk.
    Each region in a chunk will have its tokens_positive field adjusted to match the chunk's caption.
    """
    if threshold is None:
        threshold = random.randint(150, 255)

    regions_orig = result['grounding']['regions']
    chunks = []
    region_idx = 0
    total_regions = len(regions_orig)

    while region_idx < total_regions:
        temp_caption = []
        temp_regions = []
        cursor_chunk = 0

        while region_idx < total_regions:
            region = regions_orig[region_idx]
            text = region['phrase']
            next_caption = '. '.join(temp_caption + [text]) + '. '
            if len(next_caption) > threshold and temp_regions:
                break

            # Character-based span
            span_start = cursor_chunk
            span_end = cursor_chunk + len(text) - 1

            chunk_region = {
                'phrase': text,
                'bboxes': region['bboxes'],
                'tokens_positive': [[span_start, span_end]],
            }
            temp_caption.append(text)
            temp_regions.append(chunk_region)

            # Advance by text length + separator (". ")
            cursor_chunk += len(text) + 2
            region_idx += 1

        chunks.append({
            'filename': result['filename'],
            'height': result['height'],
            'width': result['width'],
            'grounding': {
                'caption': '. '.join([r['phrase'] for r in temp_regions]) + '. ',
                'regions': temp_regions
            }
        })

    return chunks
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


# generate_referring_expressions_odvg.py
"""
Generate attribute-, spatial- and reasoning-based referring expressions (single, multi, non) for a synthetic dataset
and write the results to OVDG-style jsonl.

Key design changes vs. original script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **One LLM call per image** – All nine buckets of expressions are requested in **one** chat completion by
   asking the model for a structured JSON block. This halves token usage and latency.
2. **Configurable question counts** –  Numbers per bucket are read from a JSON/YAML config or CLI flags.
3. **Cleaner prompt & examples** –  Prompt now includes user-provided examples, and explicit instructions to:
     - Use `category`, `short_phrase`, and `features` fields when generating expressions.
     - Infer spatial relationships strictly from the provided `bbox` values (absolute and relative positions).
4. **Direct OVDG output** –  Each question becomes an OVDG *region* entry and grouped under one
   caption per image. `tokens_positive` are computed automatically.
5. **Stateless retries** –  Automatic validation + adaptive temperature retry loop keeps the pipeline robust.
"""

import argparse, json, re, time
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
import openai
import random
from pydantic import BaseModel, Field
from typing import List
import os
from multiprocessing import Pool
import tempfile
from google.cloud import storage

# --- GCS helpers ---
def parse_gcs_path(gcs_path: str):
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Not a valid GCS path: {gcs_path}")
    path = gcs_path[5:]
    bucket, _, blob = path.partition("/")
    return bucket, blob

def download_from_gcs(gcs_path: str, local_path: str):
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_path} to {local_path}")

def upload_to_gcs(local_path: str, gcs_path: str):
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to {gcs_path}")

class Expression(BaseModel):
    q: str = Field(..., description="The referring‐expression string")
    ids: List[int] = Field(..., description="List of segment IDs referenced")

class ObjectExpressions(BaseModel):
    attribute: List[Expression] = Field(..., description="Attribute‐based expressions")
    spatial:   List[Expression] = Field(..., description="Spatial‐based expressions")
    reasoning: List[Expression] = Field(..., description="Reasoning‐based expressions")

class RefExpressions(BaseModel):
    single_object: ObjectExpressions = Field(..., description="Expressions targeting exactly one object")
    multi_object:  ObjectExpressions = Field(..., description="Expressions targeting multiple objects")

#####################################################################
# -------------------------- CONFIG --------------------------------#
#####################################################################
DEFAULT_NUM_Q = {
    "single": {"attribute": 6, "spatial": 6, "reasoning": 6},
    "multi":  {"attribute": 3, "spatial": 3, "reasoning": 3},
}

MAX_RETRIES = 10
SLEEP_BETWEEN_RETRIES = 10  # seconds

#####################################################################
# ----------------------- EXAMPLES --------------------------------#
#####################################################################
EXAMPLE_SEGMENTS = '''
[ID: 7377552] tongs | short_phrase: tongs with rough iron texture | features: [] | description: tongs with a rough iron texture, painted in old bronze | bbox: [318, 535, 128, 282]
[ID: 10569372] bath_towel | short_phrase: bath towel with tribal flair | features: [] | description: a bath_towel with geometric tribal flair in coppery tones | bbox: [474, 10, 493, 879]
[ID: 2187630] Canned | short_phrase: cylindrical can of recycled aluminum | features: [] | description: A cylindrical can made of recycled aluminum. | bbox: [376, 217, 102, 247]
[ID: 10733385] shovel | short_phrase: shovel with gleaming blade | features: [] | description: a shovel with a blade that gleams like polished alabaster | bbox: [424, 726, 48, 184]
[ID: 519546] knitting_needle | short_phrase: knitting needle with glossy finish | features: [] | description: a knitting_needle with a glossy, clear finish and a spiral ridge | bbox: [0, 80, 97, 125]
[ID: 4995055] strap | short_phrase: slim clear strap with blue stripe | features: [] | description: a slim, clear strap with a spray-painted blue stripe | bbox: [725, 178, 54, 60]
[ID: 9339368] teakettle | short_phrase: teakettle with glass body | features: [] | description: A teakettle with a round glass body and a charming, twisted copper handle. | bbox: [324, 789, 103, 117]
[ID: 8109537] cushion | short_phrase: hunter green leather cushion | features: [] | description: A hunter green, sleek leather cushion. | bbox: [684, 219, 89, 79]
[ID: 4123758] raspberry | short_phrase: glossy raspberry with maroon tinge | features: [] | description: a glossy raspberry with a subtle maroon tinge | bbox: [183, 324, 134, 140]
[ID: 9903309] dropper | short_phrase: dropper with matte black body | features: [] | description: A dropper with a matte black body and a glossy dropper tip | bbox: [204, 868, 93, 122]
[ID: 2998739] snowmobile | short_phrase: snowmobile with white surface | features: [] | description: a snowmobile with a glossy white surface decorated with longitudinal red strips | bbox: [898, 451, 63, 38]
[ID: 13570439] box | short_phrase: box with bold stripes and smiley | features: [] | description: a box painted in bold stripes with a quirky smiley face | bbox: [305, 837, 46, 42]
[ID: 232766] ram_animal | short_phrase: compact ram with patchwork wool | features: [] | description: a compact ram boasting an intricate pattern of color on its wool, resembling patchwork | bbox: [152, 704, 35, 36]
[ID: 15801147] birthday_card | short_phrase: cheerful pirate ship with map | features: [] | description: A birthday_card featuring a cheerful pirate ship with a colorful map. | bbox: [34, 924, 45, 46]
[ID: 8631767] Egg_tart | short_phrase: marigold custard egg tart | features: [] | description: An egg tart with a marigold-colored custard that slightly spills over around the cornflower-blue border. | bbox: [442, 207, 43, 44]
[ID: 12491521] cornbread | short_phrase: crispy cornbread with golden flecks | features: [] | description: A crispy slice of cornbread, with a myriad of shimmering, golden flecks and patches. | bbox: [191, 621, 42, 42]
[ID: 7054199] wooden_spoon | short_phrase: stout wooden spoon for dough | features: [] | description: a stout, thick wooden_spoon with a hefty feel, perfect for tackling hefty dough mixtures | bbox: [207, 292, 46, 40]
[ID: 2379817] motor | short_phrase: tiny pink motor with meta plates | features: [] | description: a tiny, delicate motor painted in a pastel pink with tiny meta plates | bbox: [95, 812, 40, 25]
'''

EXAMPLE_PROMPT = '''
{
  "single_object": {
    "attribute": [
      { "q": "The glossy raspberry with maroon tinge", "ids": [4123758] },
      { "q": "The slim clear strap with blue stripe", "ids": [4995055] },
      { "q": "The cylindrical can of recycled aluminum", "ids": [2187630] },
      { "q": "The hunter green leather cushion", "ids": [8109537] }
    ],
    "spatial": [
      { "q": "The knitting needle with glossy finish on the far left", "ids": [519546] },
      { "q": "The snowmobile with white surface on the far right", "ids": [2998739] }
    ],
    "reasoning": [
      { "q": "The teakettle with glass body below the shovel with the gleaming blade", "ids": [9339368] },
      { "q": "The box with bold stripes and smiley to the left of the snowmobile with white surface", "ids": [13570439] },
      { "q": "The glossy raspberry with maroon tinge to the left of the cylindrical can of recycled aluminum", "ids": [4123758] },
      { "q": "The slim clear strap with blue stripe on the bath towel with tribal flair", "ids": [4995055] },
      { "q": "The stout wooden spoon for dough above the shovel with the gleaming blade", "ids": [7054199] }
    ]
  },
  "multi_object": {
    "attribute": [
      { "q": "All the objects with a glossy finish", "ids": [519546, 4123758] },
      { "q": "All the striped objects", "ids": [13570439, 2998739] },
      { "q": "All the containers", "ids": [2187630, 13570439] },
    ],
    "spatial": [
      {
        "q": "All the objects above the horizontal midpoint of the image",
        "ids": [10569372, 2187630, 4995055, 8109537, 4123758, 8631767, 7054199, 519546]
      },
      {
        "q": "All the objects that span the central vertical band of the image",
        "ids": [7054199, 4123758, 2187630, 7377552, 2998739]
      },
      {
        "q": "All the objects to the left of center and above the shovel with gleaming blade",
        "ids": [7377552, 2187630, 519546, 4123758, 7054199, 13570439, 232766, 2379817, 8631767]
      },
      {
        "q": "All the objects surrounding the cylindrical can of recycled aluminum",
        "ids": [7377552, 7054199, 8631767, 4123758]
      }
    ],
    "reasoning": [
      { "q": "All the metallic objects to the left of the bath towel with tribal flair", "ids": [7377552, 2187630, 2379817] },
      { "q": "All the objects with a glossy finish above the box with bold stripes and smiley", "ids": [519546, 4123758] },
      { "q": "All the containers to the right of the knitting needle with glossy finish", "ids": [2187630, 13570439] },
    ]
  }
}
'''

#####################################################################
# ----------------------- PROMPT UTILS -----------------------------#
#####################################################################
OBJECT_TEMPLATE = "[ID: {id}] {category} | short_phrase: {short_phrase} | features: {features} | description: {description} | bbox: {bbox}"

SCHEMA_SNIPPET = '''
### JSON schema you MUST return
{
  "single_object": {"attribute": [{"q": str, "ids": [int]}], "spatial": [{"q": str, "ids": [int]}], "reasoning": [{"q": str, "ids": [int]}]},
  "multi_object":  { … same keys, but ids lists contain 2+ ints … }
}
'''


def build_prompt(segments: List[Dict[str, Any]], num_q: Dict[str, Dict[str, int]]) -> str:
    # Build object table
    table = "\n".join(OBJECT_TEMPLATE.format(**seg) for seg in segments)
    # Build count instructions
    counts_text = "\n".join(f"- {scope}/{kind}: {cnt} expressions" for scope, kinds in num_q.items() for kind, cnt in kinds.items())

    # Combine prompt
    prompt = f"""
You are a referring expression detection data generator. I will provide you with a list of objects WITHIN an IMAGE, and you will generate referring expressions similar to RefCOCO, RefCOCO+, RefCOCOg, and GrefCOCO.

We categorized referring expressions into 3 types: attribute-based, spatial-based, and reasoning-based.
**Requirements**:
- Attribute-based: ask about `features`, `category`, or `short_phrase` of exactly one object. (e.g. The white dog)
- Spatial-based: infer absolute or relative positions strictly from the `bbox` values (e.g. left/right, above/below, center). (e.g. The dog left to the people with brown shirt)
- Reasoning-based: combine features, short_phrase, category and spatial bbox relationships between objects. (e.g. The whilte animal left to the person with brown shirt)
- Use `short_phrase` or `features` preferentially to refer to objects; also can use `category` with some features to refer it.
- Return ONE JSON block matching the schema exactly, with **exactly** the requested counts per bucket. No extra keys.

For each referring expressions, we have 3 types of returning objects:
1. **Single object**: Expression refers to exactly one object in the image. (e.g. The white dog)
2. **Multi-object**: Expression refers to 2 or more objects in the image. (e.g. All the white dog in the images)

We have categorized referring expressions into three types: attribute-based, spatial-based, and reasoning-based.

Requirements:
- Attribute-based: Refer to exactly one object by its features, category, or short_phrase (e.g., "the white dog").
- Spatial-based: Infer absolute or relative positions strictly from bbox values (e.g., "the dog to the left of the person with a brown shirt").
- Reasoning-based: Combine features, short_phrases, categories, and spatial bbox relationships between objects (e.g., "the white animal to the left of the person wearing a brown shirt").
- Use short_phrase or features preferentially to refer to objects; you may also use category together with features.
- Return a single JSON block matching the schema exactly, containing exactly the requested counts per bucket; include no extra keys.

For each referring expression, there are three reference types:
1.  Single-object: The expression refers to exactly one object in the image (e.g., "the white dog").
2.	Multi-object: The expression refers to two or more objects in the image (e.g., "all the white dogs in the image").



# ### Example segments and expressions
# Example annotation 
{EXAMPLE_SEGMENTS}

# Example generated expression
{EXAMPLE_PROMPT}


### Here is the objects in the image (Bounding Box is provided in XYWH COCO format, you should compare then in coco's way):
{table}

### Counts to generate
{counts_text}

{SCHEMA_SNIPPET}
"""
    return prompt


#####################################################################
# ----------------------- OVDG HELPERS -----------------------------#
#####################################################################

def ovdg_from_expressions(image: Dict[str, Any], qs: Dict[str, Any]) -> Dict[str, Any]:
    regions, caption_parts, cursor = [], [], 0
    # Map segment IDs to their bounding boxes
    id2bbox = {seg['id']: seg['bbox'] for seg in image['segments_info']}

    def add_region(text: str, ids: List[int]):
        nonlocal cursor
        start, end = cursor, cursor + len(text) - 1
        # Gather all bboxes for each refZrenced ID
        bboxes = [id2bbox.get(i, [0, 0, 0, 0]) for i in ids]
        regions.append({
            'bboxes': bboxes,
            'phrase': text,
            'tokens_positive': [[start, end]]
        })
        caption_parts.append(text)
        cursor += len(text)

    # Process both single-object and multi-object queries
    for scope in ('single_object', 'multi_object'):
        for kind in ('attribute', 'spatial', 'reasoning'):
            for e in qs[scope][kind]:
                # Strip punctuation and add region
                add_region(e['q'].rstrip('?. '), e['ids'])

    return {
        'filename': image.get('file_name', ''),
        'height': image.get('height', 1024),
        'width': image.get('width', 1024),
        'grounding': {
            'caption': '. '.join(caption_parts) + '. ',
            'regions': regions
        }
    }


#####################################################################
# -------------------- MAIN PIPELINE -------------------------------#
#####################################################################
def infer_expressions(client: openai.Client, prompt: str) -> Dict[str, Any]:
    from pydantic import ValidationError

    attempt = 0
    parsed: RefExpressions | None = None

    while attempt < MAX_RETRIES:
        try:
            # resp = client.beta.chat.completions.parse(
            #     model=MODEL_NAME,
            #     messages=[{'role':'user','content':prompt}],
            #     response_format=RefExpressions
            # )
            resp = client.beta.chat.completions.parse(
                model="Qwen/QwQ-32B-AWQ",
                messages=[{'role':'user','content':prompt}],
                response_format=RefExpressions,
                temperature=0.1
            )
            # .choices[0].message.parsed is already a RefExpressions instance
            parsed = resp.choices[0].message.parsed  
            break
        except ValidationError as ve:
            # JSON structure didn’t match—retry
            attempt += 1
            print(f"[Attempt {attempt}] validation error: {ve}")
            time.sleep(SLEEP_BETWEEN_RETRIES)
        except Exception as e:
            # LLM or network error—also retry
            attempt += 1
            print(f"[Attempt {attempt}] error: {e}")
            time.sleep(SLEEP_BETWEEN_RETRIES)

    if parsed is None:
        # Give back an empty—but correctly typed—fallback
        parsed = RefExpressions(
            single_object=ObjectExpressions(attribute=[], spatial=[], reasoning=[]),
            multi_object= ObjectExpressions(attribute=[], spatial=[], reasoning=[])
        )

    # Finally return a plain dict for the rest of your pipeline
    return parsed.dict()
    






# -------------------- Distributed Runner --------------------#

def create_client(api_key: str):
    return openai.Client(base_url="http://localhost:8080/v1", api_key=api_key)

def process_annotation(args):
    idx, ann, cache_dir, api_key, orig_output_dir = args
    cache_file = os.path.join(cache_dir, f"{idx}.json")
    output_dir = orig_output_dir
    cache_dir_name = os.path.basename(cache_dir)
    if cache_dir_name.startswith("cache_job_"):
        job_index = cache_dir_name[len("cache_job_") :]
    else:
        job_index = "0"
    remote_cache_path = os.path.join(orig_output_dir, f"cache_job_{job_index}", f"{idx}.json")
    # If output_dir is a GCS path, check remote cache existence
    if output_dir.startswith("gs://"):
        bucket_name, blob_name = parse_gcs_path(remote_cache_path)
        client_gcs = storage.Client()
        bucket = client_gcs.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if blob.exists():
            # Remote cache exists, download and use it
            blob.download_to_filename(cache_file)
            with open(cache_file, 'r') as f:
                raw = json.load(f)
                chunks = chunk_result(raw)
                return idx, chunks
    # Only check local cache for non-GCS cases
    if not output_dir.startswith("gs://") and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        with open(cache_file, 'r') as f:
            raw = json.load(f)
            chunks = chunk_result(raw)
            return idx, chunks

    client = create_client(api_key)
    prompt = build_prompt(ann['segments_info'], DEFAULT_NUM_Q)

    # Use the shared infer_expressions function instead of inline logic
    qs = infer_expressions(client, prompt)

    valid_ids = {seg['id'] for seg in ann['segments_info']}
    for scope in ('single_object','multi_object'):
        for kind in ('attribute','spatial','reasoning'):
            for expr in qs[scope][kind]:
                expr['ids'] = [rid for rid in expr['ids'] if rid in valid_ids]

    result = ovdg_from_expressions(ann, qs)
    chunks = chunk_result(result)
    with open(cache_file, 'w') as f:
        json.dump(result, f)
    # After writing local cache, upload to GCS if output_dir is GCS
    if output_dir.startswith("gs://"):
        upload_to_gcs(local_path=cache_file, gcs_path=remote_cache_path)
    return idx, chunks

def main():
    parser = argparse.ArgumentParser(description="Distributed ODVG inference")
    parser.add_argument('total_jobs', type=int, help='Total number of jobs')
    parser.add_argument('job_index', type=int, help='Index of this job (0-based)')
    parser.add_argument('input_file', type=str, help='Path to input JSONL annotations')
    parser.add_argument('output_dir', type=str, help='Directory for outputs and cache')
    parser.add_argument('--api_key', type=str, default="placeholder", help='OpenAI API key')
    parser.add_argument('--num_workers', type=int, default=300, help='Number of parallel processes')
    args = parser.parse_args()

    # --- GCS input/output support ---
    # Download input file if on GCS
    is_gcs_input = args.input_file.startswith("gs://")
    if is_gcs_input:
        local_input = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl").name
        download_from_gcs(args.input_file, local_input)
        args.input_file = local_input

    # Prepare local output directory (and flag for later upload)
    is_gcs_output = args.output_dir.startswith("gs://")
    if is_gcs_output:
        local_output_dir = tempfile.mkdtemp()
    else:
        local_output_dir = args.output_dir

    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY")

    os.makedirs(local_output_dir, exist_ok=True)
    cache_dir = os.path.join(local_output_dir, f"cache_job_{args.job_index}")
    os.makedirs(cache_dir, exist_ok=True)

    annotations = []
    annotations = json.load(open(args.input_file)).get('annotations', [])

    total = len(annotations)
    per_job = total // args.total_jobs
    start = args.job_index * per_job
    end = total if args.job_index == args.total_jobs-1 else start + per_job
    partition = annotations[start:end]

    tasks = [(i, ann, cache_dir, api_key, args.output_dir) for i, ann in enumerate(partition)]
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_annotation, tasks), total=len(tasks)))

    # results is a list of (idx, chunks), where chunks is a list of dicts
    results.sort(key=lambda x: x[0])
    output_path = os.path.join(local_output_dir, f"job_{args.job_index}.jsonl")
    with open(output_path, 'w') as out_f:
        for _, chunks in results:
            if chunks is None:
                continue
            for entry in chunks:
                out_f.write(json.dumps(entry) + '\n')

    # print(f"Job {args.job_index} complete: wrote {len(results)} records to {output_path}")

    # Upload result back to GCS if requested
    if is_gcs_output:
        gcs_dest = args.output_dir.rstrip("/") + f"/job_{args.job_index}.jsonl"
        upload_to_gcs(output_path, gcs_dest)

if __name__ == '__main__':
    main()
