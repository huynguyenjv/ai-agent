# Opt-in Multi-Agent (`/agents`) — Design

> **Trạng thái:** THIẾT KẾ (chưa implement). Viết 2026-06-05.
> **Quyết định:** kích hoạt bằng **slash-prefix `/agents`**; mặc định vẫn single-agent.
> **Liên quan:** improvement-plan Phase 15 (bản đầy đủ); tài liệu này là **bản nhẹ, in-process**, không dùng Redis/message bus.

---

## 1. Mục tiêu & nguyên tắc

- **Mặc định = single-agent** (graph hiện tại): nhanh, rẻ, không đổi hành vi.
- **Opt-in per-request**: khi user gõ `/agents <yêu cầu>` → chạy **multi-agent workflow** cho riêng request đó. Không có `/agents` → single-agent như cũ.
- **Không hạ tầng mới**: multi-agent chạy **in-process** bằng một LangGraph lớn hơn. **Không** Redis, **không** message bus, **không** agent ở process riêng (đó là Phase 15 đầy đủ — chưa cần).
- **Agentic-first**: agent "Researcher" thu thập context bằng tool client-side (`vtrip_grep`/`vtrip_search_symbol`/`vtrip_read_file`), **không** dùng RAG/Qdrant (đang off).
- **Forward-compatible với CLI**: CLI chỉ cần chèn prefix `/agents` (hoặc cờ `--agents` → tự chèn).

---

## 2. Kích hoạt: slash-prefix `/agents`

### Quy tắc parse
- Xét **message user mới nhất**. Nếu nội dung (sau khi trim) **bắt đầu** bằng token `/agents` (case-insensitive), theo sau là khoảng trắng/xuống dòng hoặc hết chuỗi:
  - đặt `state["multi_agent"] = True`
  - **strip** token `/agents` khỏi query, phần còn lại là yêu cầu thật.
- Chỉ xét turn mới nhất → không "dính" từ turn cũ (trừ khi muốn persist qua session — xem §7).
- Vị trí parse: `server/routers/chat.py` (trước khi build state/graph), tách thành helper `parse_agent_directives(text) -> (clean_text, multi_agent: bool)`.

### Lưu ý client
- Một số IDE (Continue/Tabby) **có thể tự bắt ký tự `/`** làm slash-command của IDE. Nếu bị nuốt, dự phòng token thay thế: `@agents` hoặc field/header (xem §8 "mở rộng"). → cần test thực tế trên Continue khi implement.

---

## 3. Thay đổi state

`AgentState` (`server/agent/state.py`) thêm:
```python
multi_agent: bool                 # request chạy multi-agent workflow
agent_role: str                   # role đang chạy (planner/researcher/coder/reviewer)
research_context: str             # output Researcher gom được
coder_output: str                 # draft của Coder
reviewer_feedback: str            # feedback của Reviewer
agent_iterations: int             # đếm vòng lặp Coder↔Reviewer (giới hạn)
```

---

## 4. Chọn graph (điểm rẽ duy nhất)

`server/routers/chat.py`:
```python
from server.agent.graph import build_agent_graph
from server.agent.multi_agent_graph import build_multi_agent_graph   # NEW

if multi_agent:
    agent = build_multi_agent_graph(vllm_client, model, sse_callback=sse_callback)
else:
    agent = build_agent_graph(...)   # như hiện tại
```
Single-agent giữ **nguyên xi**. Mọi thứ multi-agent nằm trong file mới `server/agent/multi_agent_graph.py`.

---

## 5. Multi-agent graph (in-process LangGraph)

```
classify_intent → planner ─▶ researcher ─▶ coder ─▶ reviewer
                                                       │
                              ┌── approved ───────────┘
                              ▼                         └── changes_requested ─▶ coder (loop ≤ N)
                         post_process → END
```

| Role (node) | Trách nhiệm | Tái dùng |
|-------------|-------------|----------|
| **Planner** | Phân rã yêu cầu thành các bước + xác định context cần | `planner.py` hiện có (mở rộng prompt) |
| **Researcher** | Gom context bằng agentic tools (grep/search/read), tóm tắt vào `research_context` | tool-call loop + `summarize.py` |
| **Coder** | Sinh/sửa code theo plan + research_context | `generate.py` (prompt role "coder") |
| **Reviewer** | Chấm chất lượng/bảo mật, trả `approved` hoặc feedback | `critic.py` (mở rộng) |
| **Executor** *(tùy chọn, sau)* | Chạy test/lint qua tool, đưa kết quả về Coder | `vtrip_run_tests`/`vtrip_lint_code` |

- **Loop có giới hạn**: Coder↔Reviewer tối đa `agent_iterations` (vd 2–3) để tránh vòng lặp vô hạn.
- Mỗi node = 1 LLM call với **persona prompt riêng** (giống `INTENT_PROMPTS` hiện có, nhưng theo role).
- Dùng chung `sse_callback` để stream tiến trình ("🧭 Planner…", "🔎 Researcher…", "💻 Coder…", "✅ Reviewer…").

---

## 6. Cái KHÔNG làm (và vì sao)

| Bỏ | Lý do |
|----|-------|
| Redis Pub/Sub message bus (`message_bus.py`) | Chỉ cần khi agent ở process/máy khác. Đây là in-process, gọi hàm trực tiếp qua LangGraph edges. |
| `coordinator.py` riêng | LangGraph state machine **chính là** coordinator. |
| `AgentMessage`/`protocol.py` | State dùng chung thay cho message passing. |
| Cross-session memory (`memory_store.py`) | Phase 15.4 riêng; không thuộc phạm vi `/agents`. |

→ Khi nào cần agent **phân tán** (scale theo role, chạy song song đa máy) mới nâng lên Phase 15 đầy đủ.

---

## 7. Persist qua nhiều turn? (quyết định khi implement)

Hai lựa chọn:
- **(A) Per-message** (đề xuất): chỉ turn nào gõ `/agents` mới multi-agent. Đơn giản, rõ ràng.
- **(B) Sticky theo conversation**: gõ `/agents` một lần → cả hội thoại multi-agent cho tới khi `/single`. Lưu cờ vào `session.py`. Tiện hơn nhưng dễ "quên đang ở mode nào".

Mặc định nên (A).

---

## 8. Tích hợp CLI (sau này)

- CLI: `mycli chat --agents "<yêu cầu>"` → client **tự chèn** prefix `/agents ` vào message rồi gọi `/v1/chat/completions` như bình thường. **Không cần** thay đổi server API.
- Tức server chỉ cần hiểu slash-prefix; CLI/IDE/bất kỳ client nào cũng dùng được.

**Mở rộng (nếu IDE nuốt `/`):** thêm hỗ trợ field `mode:"multi"` hoặc header `X-Agent-Mode: multi` → cùng set `multi_agent=True`. Giữ slash-prefix làm chính.

---

## 9. Đánh đổi

| | Single-agent (default) | Multi-agent (`/agents`) |
|---|---|---|
| LLM calls | 1 chuỗi node | nhiều hơn (planner+researcher+coder+reviewer, có loop) |
| Độ trễ / token | Thấp | Cao hơn rõ rệt |
| Chất lượng task phức tạp | Khá | Tốt hơn (có research + review chuyên biệt) |

→ Đúng tinh thần opt-in: **chỉ trả giá khi user chủ động cần**.

---

## 10. Checklist implement (cho lúc dựng CLI)

- [ ] `parse_agent_directives()` trong `chat.py` (+ test parse: có/không prefix, case, strip, query rỗng).
- [ ] Thêm field multi-agent vào `AgentState`.
- [ ] `server/agent/multi_agent_graph.py`: `build_multi_agent_graph()` + các node role (tái dùng planner/generate/critic, thêm researcher node).
- [ ] Role prompts (planner/researcher/coder/reviewer).
- [ ] Rẽ nhánh chọn graph trong `chat.py`.
- [ ] SSE progress events theo role.
- [ ] Giới hạn vòng lặp Coder↔Reviewer.
- [ ] Tests: end-to-end multi-agent (mock vLLM), single-agent không đổi.
- [ ] (Tùy chọn) Executor node chạy test/lint.
- [ ] CLI flag `--agents` chèn prefix.

---

## 11. Tóm tắt

`/agents` = **một cờ per-request** chọn giữa hai graph trong cùng một server. Multi-agent là **một LangGraph lớn hơn** tái dùng node/tool sẵn có, **không** thêm Redis/message bus. Single-agent vẫn là mặc định nhanh-rẻ. Hợp cả IDE lẫn CLI vì chỉ là prefix text. Khi cần agent phân tán thật mới nâng Phase 15 đầy đủ.

*Tài liệu thiết kế — implement khi dựng CLI.*
