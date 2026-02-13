#!/bin/bash

# Total tasks to run (0 to 13823)
TOTAL_TASKS=13824
# Max size allowed by your cluster
CHUNK_SIZE=1000

for (( OFFSET=0; OFFSET<$TOTAL_TASKS; OFFSET+=$CHUNK_SIZE )); do
    # Calculate the end of the current chunk
    END=$(( OFFSET + CHUNK_SIZE - 1 ))

    # Ensure we don't exceed the total task count
    if [ $END -ge $TOTAL_TASKS ]; then
        END=$(( TOTAL_TASKS - 1 ))
    fi

    # Calculate the array range for this specific submission (e.g., 0-999)
    ARRAY_RANGE="0-$(( END - OFFSET ))"

    echo "Submitting array $ARRAY_RANGE with OFFSET $OFFSET..."

    # Submit and pass the OFFSET as an environment variable
    sbatch --array=$ARRAY_RANGE --export=ALL,OFFSET=$OFFSET job_submission.sh
done
