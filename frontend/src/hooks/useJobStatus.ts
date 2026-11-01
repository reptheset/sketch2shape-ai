"use client";

import { useEffect, useState } from "react";
import { ApiError, getJobStatus, type JobStatusResponse } from "@/lib/api";

export const POLL_INTERVAL_MS = 3_000;
const MAX_CONSECUTIVE_FAILURES = 5; // ~15 s of the backend being unreachable before giving up

type State = {
  status: JobStatusResponse | null;
  /** Problem talking to the backend (not a failed job; that is status.status === "failed"). */
  pollError: string | null;
};

/**
 * Polls GET /api/status/{jobId} every 3 seconds until the job completes or fails.
 * Short network blips are retried silently; a 404 (e.g. the server restarted and the
 * in-memory job is gone) stops polling immediately.
 */
export function useJobStatus(jobId: string | null): State {
  const [state, setState] = useState<State>({ status: null, pollError: null });

  useEffect(() => {
    setState({ status: null, pollError: null });
    if (!jobId) return;

    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    async function poll() {
      try {
        const status = await getJobStatus(jobId!, controller.signal);
        failures = 0;
        setState({ status, pollError: null });
        if (status.status === "completed" || status.status === "failed") return; // finished: stop polling
      } catch (err) {
        if (controller.signal.aborted) return;
        if (err instanceof ApiError && err.status === 404) {
          setState((s) => ({
            ...s,
            pollError: "This job no longer exists on the server (it may have been restarted). Please generate again.",
          }));
          return;
        }
        failures += 1;
        if (failures >= MAX_CONSECUTIVE_FAILURES) {
          setState((s) => ({
            ...s,
            pollError: err instanceof Error ? `${err.message} Is the backend still running?` : "Lost connection to the backend.",
          }));
          return;
        }
      }
      timer = setTimeout(poll, POLL_INTERVAL_MS);
    }

    poll();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  return state;
}
