"""Download an RL-trained RaR-Medicine checkpoint from S3 into ``models/<run>/``
next to this file.

Default run: ``medqa_gemma4-e2b-it_fullft`` (full fine-tune GRPO), which
``eval_test1k.py`` knows as ``--model fullft``. The earlier LoRA run is
``unmerged_model`` (a PEFT save; needs ``merge_adapter.py``). Pick another run
with ``--prefix .../output/<run>/ --dest models/<run>``.

Credentials
-----------
The machine's default role (``staging-remote-compute-role``) is *denied*
``s3:ListBucket`` / ``s3:GetObject`` on this bucket, so you need one of:

* a profile with access in ``~/.aws/credentials`` -> ``--profile NAME``
* env vars ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` [/ ``AWS_SESSION_TOKEN``]
* or ask the bucket owner to grant the role read on the prefix.

Usage::

    python download_model.py                      # default profile / env creds
    python download_model.py --profile science    # named profile
    python download_model.py --dry-run            # just list what would be fetched

Equivalent one-liner if you have the AWS CLI::

    aws s3 sync s3://oumi-science-donotdelete/shanghong/runpod-node1/oumi/experiments/rar_medicine/variant_b/output/medqa_gemma4-e2b-it_fullft/ \
        notebooks/rar_medicine_variant_b/models/medqa_gemma4-e2b-it_fullft/ --region us-west-2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

BUCKET = "oumi-science-donotdelete"
REGION = "us-west-2"
RUN = "medqa_gemma4-e2b-it_fullft"  # full-FT GRPO run; earlier LoRA run was "unmerged_model"
PREFIX = f"shanghong/runpod-node1/oumi/experiments/rar_medicine/variant_b/output/{RUN}/"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--region", default=REGION)
    p.add_argument("--profile", default=None, help="AWS profile name.")
    p.add_argument("--dest", default=str(MODELS_DIR / RUN))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that exist with matching size.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        sys.exit("pip install boto3   (not in the oumi env by default)")

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    s3 = session.client("s3")
    print(f"identity: {session.client('sts').get_caller_identity()['Arn']}")

    paginator = s3.get_paginator("list_objects_v2")
    try:
        objects = [
            o
            for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix)
            for o in page.get("Contents", [])
        ]
    except ClientError as e:
        sys.exit(
            f"{e.response['Error']['Code']}: cannot list s3://{args.bucket}/{args.prefix}\n"
            "-> your credentials lack s3:ListBucket on this bucket; see the module docstring."
        )
    if not objects:
        sys.exit("prefix is empty or does not exist")

    total = sum(o["Size"] for o in objects)
    print(
        f"{len(objects)} objects, {total / 1e9:.2f} GB under s3://{args.bucket}/{args.prefix}"
    )
    dest = Path(args.dest)
    for o in objects:
        rel = o["Key"][len(args.prefix) :]
        if not rel or rel.endswith("/"):
            continue
        target = dest / rel
        skip = (
            target.exists()
            and target.stat().st_size == o["Size"]
            and not args.overwrite
        )
        print(f"  {'skip ' if skip else 'fetch'} {rel}  ({o['Size'] / 1e6:.1f} MB)")
        if args.dry_run or skip:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(args.bucket, o["Key"], str(target))
    if not args.dry_run:
        print(f"done -> {dest}")


if __name__ == "__main__":
    main()
