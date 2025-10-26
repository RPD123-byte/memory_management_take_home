#!/usr/bin/env python3
"""
Synthetic tool-execution trace generator (cache-friendly, schema-matched).

Output shape (each item):
{
  "timestamp": "2024-01-15T10:30:00Z",
  "action_type": "...",
  "action": {...},
  "result": {"status": "success"|"error", "output": "...", "error": null|"..."},
  "context": {"reasoning": "...", "description": "..."}
}

Example:
  python scripts/synth_trace.py \
    --out examples/tool_execution_trace.json \
    --n 120 \
    --dup-rate 0.30 \
    --write-rate 0.30 \
    --error-rate 0.08 \
    --burst-prob 0.15 \
    --burst-len 3 \
    --hotset-frac 0.4 \
    --hotset-weight 0.75 \
    --seed 42
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# -----------------------
# Pools (edit as desired)
# -----------------------
FILES = [
    "app/database.py", "app/main.py", "app/routes.py",
    "infra/iam.tf", "infra/s3.tf", "scripts/cleanup.sh",
    "app/config.yaml", "app/README.md",
]
CODE_SNIPPETS = [
    "from sqlalchemy import create_engine",
    "print('hello world')",
    "resource \"aws_iam_policy\" \"example\" {}",
]
QUERIES = [
    "database connection patterns",
    "find usages of boto3",
    "search for os.getenv",
    "terraform backend config",
]
DOC_LANGS   = ["terraform", "ansible", "aws_cdk", "pulumi"]
DOC_VERS    = ["v4.22.32", "v3.5.0", "v6.5.0"]
DOC_METHODS = ["exact", "fuzzy", "hybrid"]
DOC_QUERIES = [
    "aws_iam_policy resource example",
    "terraform s3 backend docs",
    "sqlalchemy engine pool settings",
]
BUCKETS = [
    "terraform-state-demo-12db66cf",
    "thanos-metrics-dev-us-east-2-980921723213",
    "test-index-codebase",
    "my-artifacts-bucket",
    "my-bucket",
]
ACCOUNTS = ["980921723213", "123456789012"]
POLICIES = ["CustomAdministratorAccess", "ReadOnlyAccess", "S3FullAccess"]
GROUPS   = ["CustomAdministratorAccessGroup", "DataScientists", "DevOps"]

# -----------------------
# Helpers
# -----------------------
def iso_z(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"

def maybe(p: float) -> bool:
    return random.random() < p

def success(out: str = "") -> dict:
    return {"status": "success", "output": out, "error": None}

def failure(msg: str = "synthetic error") -> dict:
    return {"status": "error", "output": "", "error": msg}

def deep_clone(x):
    return json.loads(json.dumps(x))

def pick_with_hotset(all_items, hotset_frac: float, hotset_weight: float):
    n = max(1, int(len(all_items) * hotset_frac))
    hot = all_items[:n]
    cold = all_items[n:]
    if not cold:
        return random.choice(hot)
    if maybe(hotset_weight):
        return random.choice(hot)
    return random.choice(cold)

def ctx(reasoning: str, description: str) -> dict:
    return {"reasoning": reasoning, "description": description}

# -----------------------------------
# Emitters (match your action shapes)
# -----------------------------------
def em_execute_s3_ls(bucket: str) -> dict:
    cmd = f"aws s3 ls --recursive s3://{bucket}"
    out = (
        f"2025-04-09 02:19:07 terraform-elk-stack-state\n"
        f"2025-06-14 18:31:27 terraform-state-demo-12db66cf\n"
        f"2024-12-04 04:13:55 test-bucket-dcc2aa3d\n"
        f"2025-05-19 04:09:28 test-index-codebase\n"
        f"2024-11-24 05:55:43 test-yja-org-dev-serverlessdeploymentbucket-w4j7wfhqhl1a\n"
        f"2025-06-22 18:49:38 thanos-metrics-dev-us-east-2-980921723213\n"
        f"2025-05-02 18:45:03 vikram-s3-testing-ohio-us-east-2"
    )
    return {
        "action_type": "execute_command",
        "action": {"command": cmd},
        "result": success(out),
        "context": ctx("List all S3 buckets to understand current infrastructure",
                       "Executing AWS S3 list command to inventory buckets"),
    }

def em_execute_iam_groups_for_user(user: str = "sritan-iam") -> dict:
    cmd = f"aws iam list-groups-for-user --user-name {user}"
    out = (
        '{\n    "Groups": [\n        {\n'
        '            "GroupName": "CustomAdministratorAccessGroup",\n'
        '            "GroupId": "AGPA6IY35VFGSH2AYBV64",\n'
        '            "Arn": "arn:aws:iam::980921723213:group/CustomAdministratorAccessGroup",\n'
        '            "CreateDate": "2025-06-15T18:43:50+00:00"\n'
        "        }\n    ]\n}"
    )
    return {
        "action_type": "execute_command",
        "action": {"command": cmd},
        "result": success(out),
        "context": ctx("Check IAM group membership for a user",
                       "Retrieving IAM group information for user"),
    }

def em_execute_iam_list_policies_bad(group: str) -> dict:
    # malformed with &&
    cmd = f'aws iam list-attached-group-policies --group-name {group} && echo "oops"'
    return {
        "action_type": "execute_command",
        "action": {"command": cmd},
        "result": failure('/bin/sh: 1: Syntax error: "&&" unexpected'),
        "context": ctx("List policies for group (first attempt, wrong quoting)",
                       "Attempting to retrieve group policies with bad quoting"),
    }

def em_execute_iam_list_policies_ok(group: str) -> dict:
    cmd = f"aws iam list-attached-group-policies --group-name '{group}'"
    out = (
        '{\n    "AttachedPolicies": [\n        {\n'
        '            "PolicyName": "CustomAdministratorAccess",\n'
        '            "PolicyArn": "arn:aws:iam::980921723213:policy/CustomAdministratorAccess"\n'
        "        }\n    ]\n}"
    )
    return {
        "action_type": "execute_command",
        "action": {"command": cmd},
        "result": success(out),
        "context": ctx("Retry listing policies with proper quoting",
                       "Successfully retrieved group policies"),
    }

def em_execute_iam_get_policy(acct: str, pol: str) -> dict:
    arn = f"arn:aws:iam::{acct}:policy/{pol}"
    cmd = f"aws iam get-policy --policy-arn {arn}"
    out = json.dumps({
        "Policy": {
            "PolicyName": pol,
            "PolicyId": "ANPA6IY35VFGZFP4VDW45",
            "Arn": arn,
            "Path": "/",
            "DefaultVersionId": "v1",
            "AttachmentCount": 1,
            "PermissionsBoundaryUsageCount": 0,
            "IsAttachable": True,
            "CreateDate": "2025-06-15T09:16:38+00:00",
            "UpdateDate": "2025-06-15T09:16:38+00:00",
            "Tags": []
        }
    }, indent=2)
    return {
        "action_type": "execute_command",
        "action": {"command": cmd},
        "result": success(out),
        "context": ctx("Get detailed policy information",
                       f"Retrieved policy details for {pol}"),
    }

def em_execute_iam_account_summary() -> dict:
    cmd = "aws iam get-account-summary"
    out = json.dumps({
        "SummaryMap": {
            "GroupPolicySizeQuota": 5120,
            "InstanceProfilesQuota": 1000,
            "Policies": 73,
            "GroupsPerUserQuota": 10,
            "InstanceProfiles": 42,
            "AttachedPoliciesPerUserQuota": 10,
            "Users": 10,
            "PoliciesQuota": 1500,
            "Providers": 10,
            "AccountMFAEnabled": 1,
            "AccessKeysPerUserQuota": 2,
            "AssumeRolePolicySizeQuota": 2048,
            "PolicyVersionsInUseQuota": 10000,
            "GlobalEndpointTokenVersion": 1,
            "VersionsPerPolicyQuota": 5,
            "AttachedPoliciesPerGroupQuota": 10,
            "PolicySizeQuota": 6144,
            "Groups": 10,
            "AccountSigningCertificatesPresent": 0,
            "UsersQuota": 5000,
            "ServerCertificatesQuota": 20,
            "MFADevices": 10,
            "UserPolicySizeQuota": 2048,
            "PolicyVersionsInUse": 133,
            "ServerCertificates": 0,
            "Roles": 237,
            "RolesQuota": 1000,
            "SigningCertificatesPerUserQuota": 2,
            "MFADevicesInUse": 9,
            "RolePolicySizeQuota": 10240,
            "AttachedPoliciesPerRoleQuota": 10,
            "AccountAccessKeysPresent": 1,
            "AccountPasswordPresent": 1,
            "GroupsQuota": 300
        }
    }, indent=2)
    return {
        "action_type": "execute_command",
        "action": {"command": cmd},
        "result": success(out),
        "context": ctx("Get AWS account summary to understand resource usage",
                       "Retrieved comprehensive account summary"),
    }

def em_create_file(fp: str, body: str) -> dict:
    return {
        "action_type": "create_file",
        "action": {
            "file_path": fp,
            "content": body
        },
        "result": success(f"Created file: {fp}"),
        "context": ctx("Create database configuration file for the application",
                       "Created SQLAlchemy database configuration"),
    }

def em_modify_code(files: list[str], code: str, instructions: str) -> dict:
    files = sorted(files)
    return {
        "action_type": "modify_code",
        "action": {
            "code": code,
            "instructions": instructions,
            "files": files
        },
        "result": success(f"Modified file: {', '.join(files)}"),
        "context": ctx("Enhance configuration with environment-based settings",
                       "Added configuration helper function"),
    }

def em_read_file(fp: str, contents: str) -> dict:
    return {
        "action_type": "read_file_contents",
        "action": {"file_path": fp},
        "result": success(contents),
        "context": ctx("Verify the file contents", f"Read and verified {fp} contents"),
    }

def em_query_codebase(q: str, lines: list[str]) -> dict:
    body = "Found 3 relevant code snippets related to database connection patterns:\n\n" + "\n".join(lines)
    return {
        "action_type": "query_codebase",
        "action": {"query": q},
        "result": success(body),
        "context": ctx("Search for existing patterns in the codebase",
                       "Searched codebase for database connection patterns"),
    }

# -----------------------
# Sequence construction
# -----------------------
def looks_writey_exec(entry: dict) -> bool:
    if entry["action_type"] != "execute_command":
        return False
    cmd = (entry["action"].get("command") or "").lower()
    markers = [" create-", " put-", " attach-", " update-", " delete-", " remove-", " set-", " cp ", " mv ", " rm "]
    return any(m in f" {cmd} " for m in markers)

def generate_trace(
    n=50,
    dup_rate=0.25,
    write_rate=0.30,
    error_rate=0.08,
    seed=7,
    start: datetime | None = None,
    step_seconds: int = 60,
    burst_prob: float = 0.15,
    burst_len: int = 3,
    hotset_frac: float = 0.4,
    hotset_weight: float = 0.75,
    read_after_write_prob: float = 0.25,
):
    random.seed(seed)
    ts = start or datetime.utcnow()
    delta = timedelta(seconds=step_seconds)

    events: list[dict] = []

    # Seed a realistic AWS/IAM + code flow similar to your example
    bucket = pick_with_hotset(BUCKETS, hotset_frac, hotset_weight)
    grp    = pick_with_hotset(GROUPS, hotset_frac, hotset_weight)
    acct   = random.choice(ACCOUNTS)
    pol    = random.choice(POLICIES)

    seed_flow = [
        em_execute_s3_ls(bucket),
        em_execute_iam_groups_for_user("sritan-iam"),
        em_execute_iam_list_policies_bad(grp),
        em_execute_iam_list_policies_ok(grp),
        em_execute_iam_get_policy(acct, pol),
        em_execute_iam_account_summary(),
        em_create_file(
            "app/database.py",
            "from sqlalchemy import create_engine\nfrom sqlalchemy.ext.declarative import declarative_base\n"
            "from sqlalchemy.orm import sessionmaker\nimport os\n\n"
            "# Environment variable DATABASE_URL should be set, e.g. \"postgresql://user:password@localhost/dbname\"\n"
            "DATABASE_URL = os.getenv(\"DATABASE_URL\", \"sqlite:///./test.db\")\n\n"
            "engine = create_engine(\n"
            "    DATABASE_URL,\n"
            "    connect_args={\"check_same_thread\": False} if DATABASE_URL.startswith(\"sqlite\") else {}\n"
            ")\n\n"
            "SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\n\n"
            "Base = declarative_base()\n\n"
            "# Dependency\n"
            "def get_db():\n"
            "    db = SessionLocal()\n"
            "    try:\n"
            "        yield db\n"
            "    finally:\n"
            "        db.close()"
        ),
        em_modify_code(
            ["app/database.py"],
            "import os\nfrom typing import Dict, Any\n\n"
            "def get_database_config() -> Dict[str, Any]:\n"
            "    \"\"\"Get database configuration from environment\"\"\"\n"
            "    return {\n"
            "        \"url\": os.getenv(\"DATABASE_URL\", \"sqlite:///./test.db\"),\n"
            "        \"echo\": os.getenv(\"DATABASE_ECHO\", \"false\").lower() == \"true\",\n"
            "        \"pool_size\": int(os.getenv(\"DATABASE_POOL_SIZE\", \"5\")),\n"
            "        \"max_overflow\": int(os.getenv(\"DATABASE_MAX_OVERFLOW\", \"10\"))\n"
            "    }",
            "Add configuration helper function to the database module"
        ),
        em_read_file(
            "app/database.py",
            "from sqlalchemy import create_engine\nfrom sqlalchemy.ext.declarative import declarative_base\n"
            "from sqlalchemy.orm import sessionmaker\nimport os\nfrom typing import Dict, Any\n\n"
            "# Environment variable DATABASE_URL should be set, e.g. \"postgresql://user:password@localhost/dbname\"\n"
            "DATABASE_URL = os.getenv(\"DATABASE_URL\", \"sqlite:///./test.db\")\n\n"
            "engine = create_engine(\n"
            "    DATABASE_URL,\n"
            "    connect_args={\"check_same_thread\": False} if DATABASE_URL.startswith(\"sqlite\") else {}\n"
            ")\n\n"
            "SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\n\n"
            "Base = declarative_base()\n\n"
            "# Dependency\n"
            "def get_db():\n"
            "    db = SessionLocal()\n"
            "    try:\n"
            "        yield db\n"
            "    finally:\n"
            "        db.close()\n\n"
            "def get_database_config() -> Dict[str, Any]:\n"
            "    \"\"\"Get database configuration from environment\"\"\"\n"
            "    return {\n"
            "        \"url\": os.getenv(\"DATABASE_URL\", \"sqlite:///./test.db\"),\n"
            "        \"echo\": os.getenv(\"DATABASE_ECHO\", \"false\").lower() == \"true\",\n"
            "        \"pool_size\": int(os.getenv(\"DATABASE_POOL_SIZE\", \"5\")),\n"
            "        \"max_overflow\": int(os.getenv(\"DATABASE_MAX_OVERFLOW\", \"10\"))\n"
            "    }"
        ),
        em_query_codebase(
            "database connection patterns",
            [
                "1. app/database.py - SQLAlchemy connection setup",
                "2. app/models.py - Database model definitions",
                "3. app/config.py - Database configuration management",
            ],
        ),
    ]

    # Timestamp & append the seeded flow
    for e in seed_flow:
        e["timestamp"] = iso_z(ts)
        ts += delta
        events.append(e)

    # Buffer of recent successful READs for exact duplicates
    recent_reads: list[dict] = []
    max_buf = 60

    def consider_recent_read(e: dict):
        if e["result"]["status"] != "success":
            return
        # Treat as READ if not a writey execute_command and not a write op
        is_read = (e["action_type"] != "execute_command" or not looks_writey_exec(e)) and \
                  e["action_type"] not in {"create_file", "modify_code", "delete_file"}
        if is_read:
            recent_reads.append(e)
            if len(recent_reads) > max_buf:
                recent_reads.pop(0)

    for e in seed_flow:
        consider_recent_read(e)

    # Fill the rest up to n
    def make_random_read():
        # bias toward the same resources for better cache hits
        choice = random.choice(["s3_ls", "iam_groups_user", "iam_list_ok", "iam_get_policy",
                                "account_summary", "read_file", "query_codebase"])
        if choice == "s3_ls":
            return em_execute_s3_ls(pick_with_hotset(BUCKETS, hotset_frac, hotset_weight))
        if choice == "iam_groups_user":
            return em_execute_iam_groups_for_user("sritan-iam")
        if choice == "iam_list_ok":
            return em_execute_iam_list_policies_ok(pick_with_hotset(GROUPS, hotset_frac, hotset_weight))
        if choice == "iam_get_policy":
            return em_execute_iam_get_policy(random.choice(ACCOUNTS), random.choice(POLICIES))
        if choice == "account_summary":
            return em_execute_iam_account_summary()
        if choice == "read_file":
            fp = pick_with_hotset(FILES, hotset_frac, hotset_weight)
            body = random.choice(CODE_SNIPPETS)
            return em_read_file(fp, body)
        # query_codebase
        q = pick_with_hotset(QUERIES, 0.7, 0.9)
        return em_query_codebase(q, [
            "1. app/main.py - usage",
            "2. infra/iam.tf - sample",
            "3. app/config.py - config mgmt",
        ])

    def make_random_write():
        choice = random.choice(["create", "modify"])
        if choice == "create":
            fp = pick_with_hotset(FILES, hotset_frac, hotset_weight)
            return em_create_file(fp, random.choice(CODE_SNIPPETS))
        # modify
        k = random.choice([1, 1, 2])
        files = sorted({pick_with_hotset(FILES, hotset_frac, hotset_weight) for _ in range(k)})
        code = "# synthetic refactor\n" + random.choice(CODE_SNIPPETS) + "\n"
        return em_modify_code(files, code, "synthetic refactor")

    def make_random_error():
        return em_execute_iam_list_policies_bad(pick_with_hotset(GROUPS, hotset_frac, hotset_weight))

    while len(events) < n:
        # error vs normal
        if maybe(error_rate):
            e = make_random_error()
        else:
            # duplicate vs new
            if recent_reads and maybe(dup_rate):
                e = deep_clone(random.choice(recent_reads))  # exact duplicate → same tool_key
            else:
                if maybe(write_rate):
                    e = make_random_write()
                else:
                    e = make_random_read()

        e["timestamp"] = iso_z(ts)
        ts += delta
        events.append(e)

        consider_recent_read(e)

        # burst of identical reads
        if recent_reads and maybe(burst_prob) and len(events) < n:
            src = deep_clone(random.choice(recent_reads))
            for _ in range(burst_len - 1):
                if len(events) >= n:
                    break
                b = deep_clone(src)
                b["timestamp"] = iso_z(ts)
                ts += delta
                events.append(b)
                consider_recent_read(b)

        # read-after-write to test invalidation
        if e["result"]["status"] == "success" and maybe(read_after_write_prob) and len(events) < n:
            ra = None
            if e["action_type"] == "create_file":
                ra = em_read_file(e["action"]["file_path"], random.choice(CODE_SNIPPETS))
            elif e["action_type"] == "modify_code":
                fl = e["action"].get("files") or []
                if fl:
                    ra = em_read_file(random.choice(fl), random.choice(CODE_SNIPPETS))
            elif looks_writey_exec(e):
                # not emitting a writey exec in this generator (kept simpler)
                pass
            if ra:
                ra["timestamp"] = iso_z(ts)
                ts += delta
                events.append(ra)
                consider_recent_read(ra)

    return events[:n]

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="examples/tool_execution_trace.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dup-rate", type=float, default=0.25)
    ap.add_argument("--write-rate", type=float, default=0.30)
    ap.add_argument("--error-rate", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step-seconds", type=int, default=60)
    ap.add_argument("--burst-prob", type=float, default=0.15)
    ap.add_argument("--burst-len", type=int, default=3)
    ap.add_argument("--hotset-frac", type=float, default=0.4)
    ap.add_argument("--hotset-weight", type=float, default=0.75)
    ap.add_argument("--read-after-write-prob", type=float, default=0.25)
    args = ap.parse_args()

    random.seed(args.seed)
    start = datetime(2024, 1, 15, 10, 30, 0)  # deterministic-ish anchor like your example

    trace = generate_trace(
        n=args.n,
        dup_rate=args.dup_rate,
        write_rate=args.write_rate,
        error_rate=args.error_rate,
        seed=args.seed,
        start=start,
        step_seconds=args.step_seconds,
        burst_prob=args.burst_prob,
        burst_len=args.burst_len,
        hotset_frac=args.hotset_frac,
        hotset_weight=args.hotset_weight,
        read_after_write_prob=args.read_after_write_prob,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(trace, f, indent=2)
    print(f"Wrote {len(trace)} events to {out_path}")

if __name__ == "__main__":
    main()
