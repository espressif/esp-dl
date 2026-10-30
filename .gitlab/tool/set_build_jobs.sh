#!/usr/bin/env bash
# Compute a memory-safe ninja parallelism for idf.py and export IDF_PY_BUILD_JOBS.
#
# Ninja sizes its job pool from the CPU count it can see, which on a CI runner is
# the host's core count, not the per-container memory cap. On a 32-core host each
# build would then run ~34 concurrent cc1plus processes, which overruns a 32 GB
# container long before it runs out of CPU. Derive the job count from the cgroup
# memory limit instead and let the CPU count only act as an upper bound.
#
# Source this file (". .gitlab/tool/set_build_jobs.sh"), do not execute it.
#
# Tunables (CI variables):
#   IDF_PY_BUILD_JOBS  - if already set, it is respected and nothing is computed
#   BUILD_MEM_PER_JOB_MB - assumed peak RSS of one compiler process (default 2048)
#   BUILD_MEM_RESERVE_MB - memory left for ninja/python/ccache etc. (default 2048)
#   BUILD_MAX_JOBS       - hard upper bound (default 16)

if [ -n "${IDF_PY_BUILD_JOBS}" ]; then
    echo "IDF_PY_BUILD_JOBS already set to ${IDF_PY_BUILD_JOBS}, keeping it"
    return 0 2>/dev/null || exit 0
fi

mem_per_job_mb=${BUILD_MEM_PER_JOB_MB:-2048}
mem_reserve_mb=${BUILD_MEM_RESERVE_MB:-2048}
max_jobs=${BUILD_MAX_JOBS:-16}

# Memory budget: cgroup v2, then cgroup v1, then whatever the host reports.
mem_limit_mb=0
if [ -r /sys/fs/cgroup/memory.max ]; then
    v=$(cat /sys/fs/cgroup/memory.max)
    [ "$v" != "max" ] && mem_limit_mb=$((v / 1024 / 1024))
elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    v=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    # An "unlimited" v1 cgroup reports a value close to 2^63.
    [ "$v" -lt 1000000000000 ] && mem_limit_mb=$((v / 1024 / 1024))
fi
if [ "$mem_limit_mb" -eq 0 ]; then
    mem_limit_mb=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
fi

# CPU budget: a cpu.max/cfs quota is the real core count, nproc is the host's.
cpu_jobs=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    [ "$quota" != "max" ] && cpu_jobs=$(((quota + period - 1) / period))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$quota" -gt 0 ] && cpu_jobs=$(((quota + period - 1) / period))
fi

mem_jobs=$(((mem_limit_mb - mem_reserve_mb) / mem_per_job_mb))
jobs=$mem_jobs
[ "$cpu_jobs" -lt "$jobs" ] && jobs=$cpu_jobs
[ "$max_jobs" -lt "$jobs" ] && jobs=$max_jobs
[ "$jobs" -lt 1 ] && jobs=1

export IDF_PY_BUILD_JOBS=$jobs
echo "Build parallelism: IDF_PY_BUILD_JOBS=${IDF_PY_BUILD_JOBS}" \
     "(memory ${mem_limit_mb} MB -> ${mem_jobs} jobs, cpu -> ${cpu_jobs} jobs, cap ${max_jobs})"
