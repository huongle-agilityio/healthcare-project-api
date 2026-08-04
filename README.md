# Agent Graph Architecture

## TL;DR

- **Pattern**: supervisor/router — `intentClassification` phân loại ý định, rồi `Command({ goto })`
  route sang 1 trong 6 node domain (weather/places/booking/policy/memory/responder). Không gộp
  thành 1 agent vì: routing kém chính xác hơn khi nhiều tool mô tả giống nhau, `bookingAgent` cần
  `interrupt()` cho human-in-the-loop, mỗi domain cần model khác nhau (`largeModel` vs
  `smallModel`), và mỗi domain có reducer state + test riêng.
- **2 kiểu node**: "function node" (không gọi LLM, chỉ plumbing — vd `load-context.ts`) vs "agent
  node" (`bindTools`, model tự quyết định gọi gì — weather/places/policy/memory).
  `booking-agent.ts` là ngoại lệ: gọi thẳng MCP tool (không `bindTools`) để kiểm soát chắc chắn
  việc pause/resume.
- **Memory**: short-term (message/state theo từng thread) = checkpointer `PostgresSaver`;
  long-term (preference/profile theo từng user) = `PostgresStore`, 2 key (`travel`, `profile`)
  trong 1 namespace. Không tự động chèn vào context — chỉ load khi `memoryAgent` gọi
  `recall_memory`.
- **DB**: 7 bảng — 6 bảng do `@langchain/langgraph-checkpoint-postgres` tự tạo (checkpoint\* +
  store), 1 bảng tự viết (`policy_chunks` cho RAG).
- **Gap đã biết**: chưa có bước trim/summarize message — `messages` phình vô hạn theo từng
  thread, chi phí tăng tuyến tính theo số turn.

## Mục lục

- [Agent Graph Architecture](#agent-graph-architecture)
  - [TL;DR](#tldr)
  - [Mục lục](#mục-lục)
  - [1. Vì sao tách nhiều node riêng thay vì 1 agent to duy nhất?](#1-vì-sao-tách-nhiều-node-riêng-thay-vì-1-agent-to-duy-nhất)
  - [2. Từng node: model, kiểu, tool](#2-từng-node-model-kiểu-tool)
  - [3. Flow của graph](#3-flow-của-graph)
  - [4. Memory: short-term vs long-term](#4-memory-short-term-vs-long-term)
  - [8. Threads — implement \& lưu ở đâu](#8-threads--implement--lưu-ở-đâu)
  - [9. RAG — pgvector vs keyword, node hay tool](#9-rag--pgvector-vs-keyword-node-hay-tool)
  - [10. Viết unit test (step by step)](#10-viết-unit-test-step-by-step)
  - [11. Time travel — fork lại 1 booking đã confirm](#11-time-travel--fork-lại-1-booking-đã-confirm)
  - [12. Đi từ đầu đến cuối 1 flow: booking khách sạn (map với target LangChain/LangGraph)](#12-đi-từ-đầu-đến-cuối-1-flow-booking-khách-sạn-map-với-target-langchainlanggraph)
  - [13. Known issue — HITL booking re-render sau mỗi lần select](#13-known-issue--hitl-booking-re-render-sau-mỗi-lần-select)
  - [14. Target checklist — LangChainJS \& LangGraph](#14-target-checklist--langchainjs--langgraph)
    - [LangChainJS](#langchainjs)
    - [LangGraph](#langgraph)

## 1. Vì sao tách nhiều node riêng thay vì 1 agent to duy nhất?

Graph là **supervisor/router**: `intentClassification` phân loại ý định rồi `Command({ goto })`
sang 1 trong 6 node domain. Gộp thành 1 agent bind hết tool sẽ gặp:

- Model khó chọn đúng tool khi có nhiều tool mô tả gần giống nhau (routing accuracy giảm).
- `bookingAgent` cần `interrupt()` để pause/resume chờ user chọn chuyến bay/khách sạn —
  khó làm an toàn trong 1 ReAct-loop tự do.
- Mỗi domain cần model khác nhau: `bookingAgent` dùng `largeModel` (trích slot chính xác hơn),
  còn lại dùng `smallModel`.
- Mỗi domain có reducer state khác nhau (`itinerary` overwrite, `messages` concat).
- Test theo từng node độc lập dễ hơn nhiều so với 1 agent khổng lồ.

## 2. Từng node: model, kiểu, tool

Phân biệt theo việc node có cần **model ra quyết định** hay không, và nếu có thì model đó bind
tool gì.

| Node                       | Model                                      | Kiểu                                                                                       | Mô tả                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load-context.ts`          | — (không gọi model)                        | Function thuần                                                                             | Pass `itinerary` qua nguyên trạng mỗi turn để đồng bộ state cho frontend — chỉ plumbing, không merge/validate gì                                                                                                                                                                                            |
| `intent-classification.ts` | `deterministicModel` (gpt-4o-mini, temp 0) | Structured output, không `bindTools`                                                       | Phân loại message vào 1 trong 7 intent (`weather`/`places`/`booking`/`cancel`/`policy`/`memory`/`unknown`) qua `IntentSchema`, rồi `Command({ goto })`                                                                                                                                                      |
| `weather-agent.ts`         | `smallModel` (gpt-4o-mini)                 | Agent node — bind `weatherTool`                                                            | Model tự quyết định có gọi tool weather không; tool gọi weather API                                                                                                                                                                                                                                         |
| `places-agent.ts`          | `smallModel` (gpt-4o-mini)                 | Agent node — bind `placesTool`                                                             | Model tự quyết định có gọi tool places không; tool gọi places API                                                                                                                                                                                                                                           |
| `policy-agent.ts`          | `deterministicModel` (gpt-4o-mini, temp 0) | Agent node — bind `policyTool`                                                             | Tool làm RAG bên trong: `textEmbeddingModel` (`text-embedding-3-small`) embed câu hỏi, pgvector cosine search trên `policy_chunks` (ngưỡng 0.5) trả chunk khớp nhất                                                                                                                                         |
| `booking-agent.ts`         | `largeModel` (gpt-4o) — chỉ để trích slot  | **Không** `bindTools` — deterministic control flow                                         | Model chỉ trích slot (`withStructuredOutput(BookingSlotsSchema)`); có đủ field thì code gọi thẳng `searchFlight`/`searchHotel` (MCP) rồi `interrupt()` chờ user chọn. `intent === 'cancel'` không gọi model — trả message cố định (huỷ thật xảy ra ở client, qua nút Cancel trên itinerary, không qua chat) |
| `memory-agent.ts`          | `smallModel` (gpt-4o-mini)                 | Agent node — bind 3 tool (`saveMemoryTool`, `recallMemoryTool`, `saveUserProfileTool`)     | Xử lý yêu cầu lưu/hỏi lại preference tường minh. Không `interrupt()`                                                                                                                                                                                                                                        |
| `memory-capture.ts`        | `deterministicModel` (gpt-4o-mini, temp 0) | Gọi model để trích xuất, nhưng **không** `bindTools` — tự gọi thẳng tool nếu trích được gì | Chạy sau weather/places/booking, trước `responder`; chỉ quét `HumanMessage` mới nhất để tìm preference/profile fact lỡ chèn vào                                                                                                                                                                             |
| `responder.ts`             | `smallModel` (gpt-4o-mini)                 | Gọi model với toàn bộ `state.messages`, không `bindTools`                                  | Viết câu trả lời cuối; không lặp lại data card đã render                                                                                                                                                                                                                                                    |

**Tool ↔ node** (6 LangChain `tool()` bind qua `bindTools` + 2 tool MCP gọi trực tiếp, không qua model):

| Tool                | Bind ở node                       | Gọi model hay gọi thẳng                                     |
| ------------------- | --------------------------------- | ----------------------------------------------------------- |
| `weather`           | `weatherAgent`                    | model chọn qua `bindTools`                                  |
| `places`            | `placesAgent`                     | model chọn qua `bindTools`                                  |
| `policy`            | `policyAgent`                     | model chọn qua `bindTools` (RAG bên trong)                  |
| `save_memory`       | `memoryAgent`, `memoryCapture`    | `memoryAgent`: model chọn; `memoryCapture`: gọi thẳng       |
| `save_user_profile` | `memoryAgent`, `memoryCapture`    | như trên                                                    |
| `recall_memory`     | `memoryAgent`                     | model chọn qua `bindTools`                                  |
| `search_flights`    | `bookingAgent` (MCP server riêng) | gọi thẳng qua MCP client — **không** qua model tool-calling |
| `search_hotels`     | `bookingAgent` (MCP server riêng) | như trên                                                    |

`search_flights`/`search_hotels` tách MCP riêng thay vì `bindTools` vì cần deterministic control

- `interrupt()` (xem `booking-agent.ts` ở trên) — MCP còn đóng vai trò service boundary tách
  biệt logic điều phối graph khỏi call API đặt vé/khách sạn.

## 3. Flow của graph

đầu tiền là sẽ load context
sau đó agent sẽ dựa theo message user hỏi để pick agent tương ứng
với agent booking thì sẽ sử dụng mcp local
và có HITL để user có thể interract
còn với agent weather/ place thì sẽ gọi thẳng api
3 agent này sẽ thông qua một lớp memory capture để check xem user có để cập đến sở thích hay tt cá nhân ko để lưu vào long memory
còn **policy agent** embed câu hỏi của user → tìm chunk đã lưu sẵn → dùng chunk đó để trả lời có trích dẫn
**memory agent** thì save/ get long memory của user
rồi tất cả sẽ về phần response để agent trả lời user
tiếp đến trl xong sẽ lưu lại ở **checkpoint** - doan nay thật ra được ghi sau mỗi bước của graph

## 4. Memory: short-term vs long-term

short-term = "cuộc chat này đang tới đâu", long-term = "user này là ai".

- **Short-term (per-thread)**: [`src/db/checkpointer.ts`](../src/db/checkpointer.ts) —
  `PostgresSaver`, giữ `messages`/state của 1 thread giữa các turn, cho phép `interrupt()`
  pause/resume.
  checkpoint: (Short-term)
- MemorySaver → PostgresSaver (hiện tại đang xài) chỉ là đổi "nơi lưu" (RAM → DB) - short memory
- lưu: toàn bộ state của graph
- lưu thread id
- conversation được lưu ở checkpoint blobs
  → nhớ 1 cuộc hội thoại đang diễn ra (theo thread_id): tin nhắn, slot đã hỏi, itinerary, điểm dừng interrupt()...

- **Long-term (per-user)**:
- Long-term (store) - lưu "travel preference" của user
- nhớ về 1 user, xuyên suốt mọi cuộc hội thoại (theo user_id): sở thích, thói quen
- Mục đích: mở thread hoàn toàn mới vẫn không phải hỏi lại những gì user từng nói ở thread cũ.
  - [`src/db/store.ts`](../src/db/store.ts) — `PostgresStore`, key theo
    namespace `['preferences', 'demo-user']` (constants trong
    [`src/constants/db.ts`](../src/constants/db.ts)). **Không tự động** chèn vào context — chỉ
    load khi `memoryAgent` gọi `recall_memory`.
- **Lưu bằng 2 đường**: `memoryAgent` (intent = memory) và `memoryCapture` (§3) — cả 2 gọi
  chung `saveMemoryTool`/`saveUserProfileTool` ([`src/tools/memory.ts`](../src/tools/memory.ts)),
  không trùng logic.
- **2 key trong 1 namespace**: `travel` (preference booking, text tích luỹ merge+dedupe qua
  [`src/memory/save-preference.ts`](../src/memory/save-preference.ts) /
  [`recall-preference.ts`](../src/memory/recall-preference.ts)) và `profile` (field riêng —
  scalar field bị ghi đè, `hobbies` là union, qua
  [`src/memory/save-profile.ts`](../src/memory/save-profile.ts) /
  [`recall-profile.ts`](../src/memory/recall-profile.ts)). `recallMemoryTool` trả cả 2 trong
  1 lần gọi.
- **Chưa có auth thật** — `DEMO_USER_ID = 'demo-user'` hardcode trong
  [`src/constants/db.ts`](../src/constants/db.ts), share chung 1 namespace. Multi-user thật cần
  thay bằng user id từ session.

## 8. Threads — implement & lưu ở đâu

2 tầng tách biệt, không tầng nào biết tầng kia lưu gì:

- **Sidebar (danh mục thread)**: client-side `localStorage`, chỉ `{ id, name }`, không lưu nội
  dung. Thread chỉ được thêm sau khi có ít nhất 1 message (lazy creation).
- **Thread đang active**: qua `useCopilotChatConfiguration` (không phải prop `threadId` trực
  tiếp) — cần thiết để `<CopilotChat>` ẩn màn hình welcome đúng lúc.
- **Nội dung hội thoại thật**: sống trong Postgres qua checkpointer, key theo `thread_id`. Đổi
  active thread = đổi `thread_id` gửi lên agent, LangGraph tự load checkpoint mới nhất.

## 9. RAG — pgvector vs keyword, node hay tool

- **Hiện tại**: thuần pgvector (cosine distance, ngưỡng `POLICY_MATCH_MAX_DISTANCE = 0.5`).
  Không có full-text/keyword search.
- **Hybrid được không?** Được (vector + full-text + rerank), nhưng chưa cần vì corpus nhỏ.
  Đánh giá "nhỏ hay lớn" dựa vào:
  - **Số chunk**: vài chục–vài trăm = nhỏ; chục nghìn–triệu = lớn.
  - **Recall thực tế**: test vài query, xem top-3/5 có ra đúng chunk không.
  - **Query có cần match chính xác mã/số** (vd "section 4.1") không — nếu có thì cần keyword
    dù corpus nhỏ hay lớn.
  - **Chi phí thêm hybrid** (2 query + merge/rerank) có đáng so với lợi ích không.
  - Corpus hiện tại: 5 section, ~4 dòng/section → quá nhỏ để vector-only bị nhầm.
- **Node hay tool?** Tool (`policyTool`) gọi từ node (`policyAgent`) — RAG là lookup 1 bước,
  không cần pause/resume nhiều bước như `bookingAgent`, nhưng model vẫn cần quyết định có gọi
  RAG hay không → đúng kiểu tool trong agent node.
- Corpus test dùng bản ngắn (`rag/corpus/booking-policy.md`, 5 section) để đỡ tốn token.

## 10. Viết unit test (step by step)

Stack: **Vitest**. Test nằm ngay cạnh code trong `__tests__/*.test.ts` (vd
`src/nodes/__tests__/weather-agent.test.ts`). Chạy bằng `pnpm --filter agent test` (hoặc
`vitest run` trong `apps/agent`).

1. **Tìm node/function cần test** và xác định dependency của nó (model, tool, module khác import
   qua barrel `@/...`).
2. **Mock dependency bằng `vi.mock`** trước khi import — mock được hoisted, nên khai báo ở đầu
   file:
   ```ts
   vi.mock('@/models', () => ({ smallModel: { bindTools: vi.fn() } }));
   vi.mock('@/tools', () => ({ weatherTool: { name: 'weather', invoke: vi.fn() } }));
   ```
   Bỏ qua bước này với function node thuần không có dependency ngoài (vd `load-context.ts`).
3. **Import module đã mock và unit cần test** sau các lệnh `vi.mock`.
4. **Dựng `baseState: TripStateType`** với mọi field khai báo tường minh (`messages: []`,
   `destination: undefined`, ... `itinerary: []`, `toolResult: undefined`) — tái sử dụng/spread
   nó mỗi test, mỗi test chỉ override phần mình cần.
5. **Reset mock trong `beforeEach`** (`mockReset()` từng mock fn) để test không rò state qua
   nhau. Với agent node, re-stub `bindTools` trả về `{ invoke: boundInvoke }` mỗi lần.
6. **Viết 1 `it(...)` cho mỗi hành vi**, không phải mỗi nhánh code — vd "gọi tool khi model yêu
   cầu" và "chỉ trả message của model khi nó hỏi lại thay vì gọi tool". Assert cả mock call
   (`toHaveBeenCalledWith`) lẫn state patch trả về (`toEqual`).
7. **Với node có `interrupt()` (`booking-agent.ts`)**: test riêng nhánh trước-interrupt và nhánh
   resume — xem `nodes/__tests__/booking-agent.test.ts` để theo pattern.
8. **Với RAG (`retrieve-policy.ts`)**: dùng corpus test bản ngắn (§9) thay vì bản đầy đủ để đỡ
   tốn token embedding.
9. Chạy `vitest run <path>` cho từng file khi đang code, rồi chạy cả suite trước khi commit.

## 11. Time travel — fork lại 1 booking đã confirm

"Time travel" = `getStateHistory()` + `updateState()` của LangGraph — duyệt lại lịch sử
checkpoint của 1 thread, chọn 1 checkpoint cũ, rồi nhánh (fork) nó thành 1 checkpoint **mới**
mà không đụng vào bản gốc. Dùng cho tính năng "đổi ý" trên 1 chuyến bay/khách sạn đã confirm
(sửa lại 1 quyết định cũ thay vì chỉ patch mảng `itinerary` đang sống).

**Nơi cơ chế này được chứng minh** — [`apps/agent/src/__tests__/time-travel.test.ts`](../src/__tests__/time-travel.test.ts).
Test này tự dựng 1 graph nhỏ (chỉ `bookingAgent`, `MemorySaver`, không Postgres/LLM thật — mock
hết) và chứng minh cơ chế từ đầu đến cuối:

1. **Đi tới trạng thái đã confirm.** `graph.invoke(...)` chạy flow booking tới `interrupt()`
   (hotel options được đưa ra), rồi `graph.invoke(new Command({ resume: { selectedId: hotelA.id } }))`
   confirm hotel A. `graph.getState(config)` lúc này cho thấy `itinerary` chứa hotel A.
2. **Duyệt lịch sử để tìm đúng checkpoint.** `for await (const snapshot of graph.getStateHistory(config))`
   duyệt qua mọi checkpoint của thread (mới nhất trước); tìm checkpoint mà
   `snapshot.values.itinerary` lần đầu chứa hotel A — không giả định đó chỉ đơn giản là "cái mới
   nhất".
3. **Fork nó.** `graph.updateState(confirmedSnapshot.config, { itinerary: [hotelBItem] })` ghi 1
   checkpoint **mới** đè lên trên checkpoint cũ đó, với hotel B thay vì hotel A. Trả về 1
   `config` mới trỏ vào nhánh vừa fork.
4. **Verify bản gốc không bị đụng vào.** `graph.getState(confirmedState.config)` (config của
   chính checkpoint gốc, không phải config đã fork) vẫn trả về hotel A — fork không mutate lịch
   sử — và các field dùng chung (`destination`, `messages`) giống hệt nhau ở cả 2 nhánh.

**Đã implement đầy đủ, end-to-end** — 3 file, theo đúng chuỗi UI → mutation → route:

1. [`apps/web/src/components/itinerary-sidebar/change-selection-dialog.tsx`](../../web/src/components/itinerary-sidebar/change-selection-dialog.tsx) —
   dialog "Change selection" mở lại đúng các option flight/hotel ban đầu cho 1 item trong
   itinerary. Options gốc được phục hồi qua
   [`findOriginalOptionsForItem`](../../web/src/utils/booking-selection.ts), quét
   `agent.messages` tìm tool-call `booking_flight_result`/`booking_hotel_result` mà
   `buildSelectionResultMessages` trong [`booking-agent.ts`](../src/nodes/booking-agent.ts) ghi
   lại lúc confirm. User pick option mới → gọi `forkSelection(oldItem, newItem)`.
2. [`apps/web/src/hooks/use-booking-fork-mutation.tsx`](../../web/src/hooks/use-booking-fork-mutation.tsx) —
   `useBookingForkMutation` làm optimistic `agent.setState(...)` (itinerary đổi ngay trên UI),
   rồi `POST /api/booking-fork` với `{ threadId, itemId, newItem }`.
3. [`apps/web/src/app/api/booking-fork/route.ts`](../../web/src/app/api/booking-fork/route.ts) —
   route thi hành đúng 3 bước mà test đã chứng minh, nhưng **qua LangGraph Server API** thay vì
   load graph object trực tiếp trong process: `apps/web` và `apps/agent` chạy 2 process tách
   biệt, web không import được graph instance, nên phải gọi qua HTTP bằng
   [`@langchain/langgraph-sdk`](https://www.npmjs.com/package/@langchain/langgraph-sdk)'s
   `Client` — trỏ vào agent server chạy qua [`langgraph.json`](../langgraph.json), server này tự
   expose REST API chuẩn LangGraph Server (cùng server mà
   [`itinerary-state/route.ts`](../../web/src/app/api/itinerary-state/route.ts) cũng gọi, nhưng
   qua raw `fetch` thay vì SDK):
   1. `client.threads.getHistory(threadId, { limit })` — duyệt lịch sử checkpoint (mới nhất
      trước), parse `itinerary` của từng checkpoint qua
      [`parseItineraryItems`](../../web/src/utils/itinerary.ts), dừng ở checkpoint **đầu tiên**
      chứa `itemId`.
   2. Nếu không tìm thấy → 404. Tìm thấy → build `nextItinerary` (itinerary của checkpoint đó,
      thay item khớp `itemId` bằng `newItem`), gọi
      `client.threads.updateState(threadId, { values: { itinerary: nextItinerary }, checkpoint: <checkpoint tìm được ở bước 1> })` —
      fork 1 checkpoint mới đè lên checkpoint đó, checkpoint mới này trở thành state mới nhất
      của thread (turn tiếp theo LangGraph tự dùng nó).

**2 điểm cần chú ý khi implement lại (rút ra từ code review khi build tính năng này):**

- `getHistory` **mặc định chỉ lấy 10 checkpoint gần nhất** (default `limit` của SDK) — phải
  truyền `limit` rõ ràng
  ([`BOOKING_FORK_HISTORY_LIMIT`](../../web/src/constants/itinerary.ts) = 1000), không thì
  booking cũ hơn 10 checkpoint (vài turn chat sau khi confirm) sẽ bị báo 404 giả dù booking đó
  vẫn còn trong lịch sử.
- Fork từ checkpoint cũ **cố ý rewind, không merge**: field nào không truyền vào `updateState`
  (vd `messages`, các item itinerary khác) sẽ lấy nguyên giá trị của checkpoint cũ đó, không
  phải state mới nhất — vì `itinerary` dùng overwrite reducer
  ([`apps/agent/src/states/trip.ts`](../src/states/trip.ts), `reducer: (_current, update) => update`).
  Item/tin nhắn thêm vào **sau** thời điểm confirm ban đầu sẽ bị bỏ lại ở nhánh cũ (vẫn còn
  nguyên, truy được qua checkpoint gốc — không mất dữ liệu) nhưng sẽ **không có** trong nhánh
  vừa fork. Đây là bản chất thật của time travel trong LangGraph (giống `git checkout <commit cũ> -b nhánh-mới`),
  đã cân nhắc và giữ nguyên — không đổi sang "patch state mới nhất" vì làm vậy sẽ không còn là
  time travel thật, chỉ là 1 bản sao khác của route `itinerary-state`.

## 12. Đi từ đầu đến cuối 1 flow: booking khách sạn (map với target LangChain/LangGraph)

1 flow thật — "book khách sạn" — chạm gần như đủ mọi target LangChainJS/LangGraph trong 1 chỗ.
Map nhanh trước, rồi đi từng bước kèm link file.

| Target                                         | Nằm ở đâu trong flow này                                                                                                           |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Agents, Models, Tools tương tác nhau           | `bookingAgent`: `largeModel` trích slot, **code** (không phải model) quyết định gọi tool `searchHotel` — cố tình không `bindTools` |
| Message template + schema validation           | `bookingSlotExtractionPrompt` (template) + `BookingSlotsSchema` / `BookingInputSchema` / `SelectionResumeSchema` (zod)             |
| Quản lý conversational context                 | `state.messages` đưa vào model; `state.destination/dates/travelers` fallback qua các turn, xoá sau khi booking xong                |
| Prebuilt/custom middleware                     | **Repo này không dùng** — nói thẳng ở phần "gap" bên dưới, không nhận vơ                                                           |
| Real-time event streaming                      | LangGraph dev server stream event qua SSE; `useInterrupt` của CopilotKit render interrupt đang chờ ngay khi nó stream về           |
| Graph-based execution                          | `graph.ts` — `StateGraph` + routing bằng `Command({ goto })`                                                                       |
| Long-running task & state recovery             | `PostgresSaver` checkpoint state sau mỗi bước — 1 interrupt đang pause vẫn sống sót qua 1 lần restart server                       |
| Interrupt & time travel                        | `interrupt()` trong `booking-agent.ts`; fork qua `getStateHistory`/`updateState` (§11)                                             |
| Integrated AI system (LangChainJS + LangGraph) | Model LangChain + schema zod + gọi tool, tất cả chạy như 1 node bên trong graph của LangGraph                                      |

**Sơ đồ** — mỗi node ghi kèm file thật thi hành bước đó:

```mermaid
flowchart TD
  U(["User: Book a hotel in Da Nang, Aug 1-5, 2 travelers"])

  subgraph AG["apps/agent (LangGraph)"]
    direction TB
    N1["1. load-context.ts + intent-classification.ts<br/>intent = booking, Command(goto: bookingAgent)"]
    N2["2. booking-agent.ts<br/>largeModel.withStructuredOutput<br/>(prompts/booking-slot-extraction.ts)"]
    N3{"3. BookingInputSchema.safeParse<br/>(schemas/booking.ts)"}
    N3a["thiếu field → hỏi lại user<br/>(booking-agent.ts)"]
    N4["4. booking-agent.ts<br/>code gọi thẳng searchHotel (không bindTools)"]
    N4a["mcp/client.ts — MCP client"]
    N4b["mcp/booking-server.ts — MCP server<br/>(subprocess riêng)"]
    N4c["mcp/booking-actions.ts<br/>apiGet() → mock booking API"]
    N5["5. booking-agent.ts<br/>promptHotelSelection → interrupt()"]
    CKPT[("db/checkpointer.ts<br/>PostgresSaver — checkpoint")]
    N7["7. resume tại interrupt()<br/>SelectionResumeSchema"]
    N8["8. memory-capture.ts"]
    N9["responder.ts — câu trả lời cuối"]
  end

  subgraph WEB["apps/web (CopilotKit)"]
    direction TB
    N6["6. use-booking-selection-interrupt.tsx<br/>useInterrupt() nhận SSE"]
    N6b["HotelResultsList.tsx"]
    P(["User chọn 1 khách sạn"])
  end

  U --> N1 --> N2 --> N3
  N3 -- thiếu field --> N3a
  N3 -- đủ field --> N4 --> N4a --> N4b --> N4c
  N4c --> N5
  N5 --> CKPT
  N5 -. SSE stream .-> N6 --> N6b --> P
  P -. resolve selectedId .-> N7 --> N8 --> N9
  N9 --> D(["Itinerary cập nhật + trả lời cuối"])
  N9 -. đổi ý sau .-> FORK["9. api/booking-fork/route.ts<br/>time travel — fork checkpoint (§11)"]
```

**Từng bước** — user gõ "Book a hotel in Da Nang, Aug 1–5, 2 travelers":

1. **Routing.** `START → `[`loadContext`](../src/nodes/load-context.ts)`→`[`intentClassification`](../src/nodes/intent-classification.ts) —
   phân loại `intent = "booking"` qua [`IntentSchema`](../src/schemas/intent.ts) (`deterministicModel`,
   structured output, không gọi tool), rồi `Command({ goto: bookingAgent })` — khai báo trong
   [`graph.ts`](../src/graph.ts).
2. **Trích slot (Model).** [`bookingAgent`](../src/nodes/booking-agent.ts) gọi
   `largeModel.withStructuredOutput(BookingSlotsSchema)` với template
   [`bookingSlotExtractionPrompt`](../src/prompts/booking-slot-extraction.ts) (1 `SystemMessage`
   - toàn bộ `state.messages` làm context) — trích `type`/`destination`/`dates`/`travelers`,
     merge với những gì còn sót lại trong state từ turn trước.
3. **Validate schema.** Vẫn trong [`booking-agent.ts`](../src/nodes/booking-agent.ts):
   [`BookingInputSchema.safeParse`](../src/schemas/booking.ts) check các field đã merge (rule
   `.refine()` của nó yêu cầu `dates.end` cho hotel) — thiếu field thì dừng ngay ở đây, hỏi lại
   user, không bao giờ đoán.
4. **Gọi tool (deterministic, không phải model chọn).** Khi đã hợp lệ, code — không phải model —
   gọi `searchHotel` (export từ [`mcp/client.ts`](../src/mcp/client.ts)). Đây **không phải** 1
   lệnh gọi hàm bình thường trong cùng process — nó băng qua ranh giới 2 process riêng biệt:
   - `client.ts` là **MCP client**, chạy trong process của agent. Tự nó **không có logic tool
     nào cả**, chỉ biết gửi `client.callTool({ name: 'search_hotels', arguments })`.
   - Lần gọi đầu tiên, `client.ts` tự spawn 1 **process con riêng** chạy
     [`mcp/booking-server.ts`](../src/mcp/booking-server.ts) (**MCP server** — nơi thật sự có
     `server.registerTool(...)`), nói chuyện với nó qua **stdio** (stdin/stdout của process con,
     không phải network) — giống browser (client) gọi web server, chỉ khác chạy local.
   - `booking-server.ts` nhận request, gọi [`mcp/booking-actions.ts`](../src/mcp/booking-actions.ts)
     → `apiGet()` → API mock thật (đường đi giống hệt `weatherTool` gọi API weather), rồi trả
     kết quả ngược lại qua stdio cho `client.ts`.

   Đây là chỗ Agent/Model/Tool tương tác duy nhất trong graph cố tình bỏ qua `bindTools` — việc duy nhất của model là trích slot ở bước 2. Kết quả trả về (`results` — danh sách
   khách sạn) chỉ nằm trong biến local của node, **chưa gửi cho user**.
5. **Interrupt — dừng chờ người.** `promptHotelSelection` trong
   [`booking-agent.ts`](../src/nodes/booking-agent.ts) gọi `interrupt()` của LangGraph, gửi option
   hotel cho client và dừng graph giữa chừng node. [`PostgresSaver`](../src/db/checkpointer.ts)
   đã checkpoint mọi thứ tới điểm này rồi — chỗ dừng (và context của chuyến đi) sẽ sống sót nếu
   agent process bị restart.
6. **Stream trực tiếp về client.** Event interrupt stream sang frontend qua cùng 1 kết nối
   (không polling) — `useInterrupt()` trong
   [`use-booking-selection-interrupt.tsx`](../../web/src/hooks/use-booking-selection-interrupt.tsx)
   nhận event và render
   [`HotelResultsList`](../../web/src/components/travel-list/HotelResultsList.tsx) như 1
   card chọn lựa sống động.
7. **Resume.** User chọn 1 khách sạn → card gọi `resolve({ selectedId })` → frontend gửi
   `graph.invoke(new Command({ resume: { selectedId } }))`, resume đúng ngay chỗ `bookingAgent`
   đang pause ở `interrupt()`, được validate bằng `SelectionResumeSchema`.
8. **Hoàn tất.** Hotel được map thành 1 item trong itinerary, `destination`/`dates`/`travelers`
   được xoá (để booking tiếp theo không liên quan không bị thừa kế nhầm — xem §5), rồi
   [`memoryCapture`](../src/nodes/memory-capture.ts) tranh thủ check turn này xem có preference
   nào lỡ chèn vào không, và [`responder`](../src/nodes/responder.ts) viết câu trả lời cuối.
9. **Bonus — time travel.** Đổi ý về đúng booking này sau đó chính là flow ở §11: fork lại
   checkpoint từ bước 5/7 qua `getStateHistory`/`updateState` thay vì chạy lại search từ đầu.

**Gap nói thẳng** (không giấu): repo này **không có middleware layer** — không có pattern
`createMiddleware`/interceptor nào trong `apps/agent/src`, nên target đó của LangChainJS không
được thể hiện ở đây. Và streaming ở bước 6 là do framework cung cấp sẵn (SSE stream riêng của
LangGraph + runtime của CopilotKit) chứ không phải code tự viết — flow này _hưởng lợi_ từ nó,
nhưng không có logic streaming tự code nào để chỉ ra là "phần implementation".

## 13. Known issue — HITL booking re-render sau mỗi lần select

`BookingInterruptWatcher` ([`use-booking-selection-interrupt.tsx`](../../web/src/hooks/use-booking-selection-interrupt.tsx))
reset `card` về `idle` mỗi khi `agent.messages` có message mới (theo dõi qua
`agent.messages.at(-1)?.id`) — mà mỗi lần resolve/resume 1 lựa chọn đều sinh message mới. Card
cũng được `key={payloadKey}` (key theo nội dung payload), nên mỗi interrupt mới là 1 key mới →
React unmount + mount lại toàn bộ card thay vì cập nhật tại chỗ. Với flow nhiều bước (vd chọn
chặng đi rồi chặng về của vé máy bay), mỗi lần select là 1 vòng full remount.

Cố ý, không phải sơ suất — comment trong file giải thích: card state phải nằm ở component con
(không phải ở `BookingSelectionProvider` cha) để tránh stale closure trong `useInterrupt`'s
`render()` (nguyên nhân gốc của bug nút bị kẹt ở "Selecting..." mãi mãi trước đây). Đánh đổi lấy
sự đúng đắn đó là re-render/remount lặp lại sau mỗi bước chọn.

## 14. Target checklist — LangChainJS & LangGraph

Map từng target chính thức sang file thật trong repo — tương tự bảng ở §12 nhưng tách riêng,
không gói gọn trong 1 flow ví dụ, và liệt kê đủ file cho từng target thay vì chỉ 1 chỗ đại diện.

### LangChainJS

- **"Grasp how Agents, Models, and Tools interact to build intelligent workflows."**
  - [`nodes/weather-agent.ts`](../src/nodes/weather-agent.ts),
    [`places-agent.ts`](../src/nodes/places-agent.ts),
    [`policy-agent.ts`](../src/nodes/policy-agent.ts),
    [`memory-agent.ts`](../src/nodes/memory-agent.ts) — agent node dùng `bindTools()`, model tự
    quyết định gọi tool nào.
  - [`nodes/booking-agent.ts`](../src/nodes/booking-agent.ts) — ngược lại: **code** (không phải
    model) quyết định gọi `searchHotel`/`searchFlight`, cố tình không `bindTools` (§2).
  - [`tools/index.ts`](../src/tools/index.ts) — barrel tool: `weather.ts`, `place.ts`,
    `policy.ts`, `memory.ts`.

- **"Design predictable responses using message templates and schema validation."**
  - [`prompts/`](../src/prompts) — message template (`booking-slot-extraction.ts`,
    `weather-agent.ts`, `responder.ts`, ...).
  - [`schemas/`](../src/schemas) — zod validation: `BookingSlotsSchema`/`BookingInputSchema`
    (`booking.ts`), `IntentSchema` (`intent.ts`), `SelectionResumeSchema`.

- **"Manage conversational context effectively."**
  - [`states/trip.ts`](../src/states/trip.ts) — `TripState`, `messages` tích luỹ qua các turn;
    `destination`/`dates`/`travelers` fallback từ turn trước, xoá sau khi booking xong (§5 nhắc
    ở §12 bước 8) để không thừa kế nhầm sang booking tiếp theo.
  - [`nodes/load-context.ts`](../src/nodes/load-context.ts) — đồng bộ `itinerary` mỗi turn.

- **"Apply prebuilt and custom middleware for enhanced control and extensibility."**
  - **Chưa áp dụng** — không có `createMiddleware`/interceptor nào trong `apps/agent/src` (đã
    nói thẳng ở §12 "Gap"). Target này chưa được thể hiện trong repo.

- **"Improve user experience through real-time event streaming."**
  - [`langgraph.json`](../langgraph.json) — agent server tự expose SSE stream chuẩn LangGraph.
  - [`nodes/responder.ts`](../src/nodes/responder.ts) — câu trả lời cuối stream token-by-token
    (khác structured-output node như `intentClassification`, không stream được — §"sao app lúc
    stream lúc không" đã giải thích trong hội thoại).
  - [`use-booking-selection-interrupt.tsx`](../../web/src/hooks/use-booking-selection-interrupt.tsx) —
    `useInterrupt()` nhận event ngay khi stream về, render card chọn lựa sống động, không polling.

### LangGraph

- **"Orchestrate multi-step logic using graph-based execution."**
  - [`graph.ts`](../src/graph.ts) — `StateGraph` + routing bằng `Command({ goto })` xuất phát từ
    `intentClassification`.

- **"Ensure long-running tasks and state recovery."**
  - [`db/checkpointer.ts`](../src/db/checkpointer.ts) — `PostgresSaver`, checkpoint sau mỗi bước
    của graph; 1 `interrupt()` đang pause sống sót qua restart agent process.

- **"Handle interrupts and time travel — debug or rewind execution flows safely."**
  - `interrupt()` trong [`nodes/booking-agent.ts`](../src/nodes/booking-agent.ts) —
    `promptHotelSelection`/`promptFlightSelection`.
  - Time travel: chứng minh cơ chế ở
    [`__tests__/time-travel.test.ts`](../src/__tests__/time-travel.test.ts), thực thi thật ở
    [`api/booking-fork/route.ts`](../../web/src/app/api/booking-fork/route.ts) — chi tiết đầy đủ
    ở §11.

- **"Build integrated AI systems — combine LangChainJS logic with LangGraph orchestration for scalable, maintainable AI solutions."**
  - Toàn bộ [`nodes/`](../src/nodes) — mỗi file là 1 chỗ model/schema/tool của LangChain chạy như
    1 node bên trong graph LangGraph; [`graph.ts`](../src/graph.ts) lắp các node đó lại thành 1
    hệ thống hoàn chỉnh.
