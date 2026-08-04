import { NextResponse } from 'next/server';
import { z } from 'zod';
import { Client } from '@langchain/langgraph-sdk';

// Constants
import { AGENT_URL, BOOKING_FORK_HISTORY_LIMIT } from '@/constants';

// Schemas
import { ItineraryItemSchema } from '@repo/schemas';

// Utils
import { parseItineraryItems } from '@/utils';

// Types
import type { ItineraryItem } from '@repo/types';

const RequestSchema = z.object({
  threadId: z.string(),
  itemId: z.string(),
  newItem: ItineraryItemSchema,
});

interface ThreadValues {
  itinerary?: unknown;
}

/**
 * "Change my mind" on an already-confirmed flight/hotel — genuine LangGraph time travel:
 * `getHistory` walks the thread's checkpoint history to find the checkpoint where the item was
 * actually confirmed (rather than assuming it's simply "the newest"), then `updateState` forks
 * a new checkpoint off of it (same mechanism as `apps/agent/src/__tests__/time-travel.test.ts`).
 * This deliberately rewinds the thread: `itinerary` uses a pure-overwrite reducer and any
 * channel not passed to `updateState` (messages, other itinerary items) carries over from that
 * *old* checkpoint, not the latest one — anything added after the original confirmation is left
 * behind on the pre-fork branch (still reachable via that checkpoint), not merged into the fork.
 * That's inherent to real time travel, not a bug to work around.
 */
export async function POST(request: Request) {
  const body: unknown = await request.json();
  const parsed = RequestSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const { threadId, itemId, newItem } = parsed.data;
  const client = new Client({ apiUrl: AGENT_URL });

  const history = await client.threads.getHistory<ThreadValues>(threadId, {
    limit: BOOKING_FORK_HISTORY_LIMIT,
  });

  let confirmedSnapshot: (typeof history)[number] | undefined;
  let confirmedItinerary: ItineraryItem[] = [];

  for (const snapshot of history) {
    const itinerary = parseItineraryItems(snapshot.values.itinerary);
    if (itinerary.some((item) => item.id === itemId)) {
      confirmedSnapshot = snapshot;
      confirmedItinerary = itinerary;
      break;
    }
  }

  if (!confirmedSnapshot) {
    return NextResponse.json({ error: 'Original booking checkpoint not found' }, { status: 404 });
  }

  const nextItinerary: ItineraryItem[] = confirmedItinerary.map((item) =>
    item.id === itemId ? newItem : item
  );

  await client.threads.updateState<{ itinerary: ItineraryItem[] }>(threadId, {
    values: { itinerary: nextItinerary },
    checkpoint: confirmedSnapshot.checkpoint,
  });

  return NextResponse.json({ ok: true });
}
