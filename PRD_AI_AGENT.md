[PRD] AI 커머스 자동화 에이전트 개발 기획서

## 1. 개요
본 프로젝트는 커머스 데이터를 API 연동 및 브라우저 오토메이션으로 관리하는 AI 에이전트를 구축하는 것입니다. 가장 큰 난관인 '컨텍스트 오염', '비용 폭발', '동시성 충돌'을 해결하기 위해 'Main Trunk & Sub-Branch' 구조와 'Task Queue' 아키텍처를 채택합니다.

## 2. 아키텍처 원칙 (Core Principles)
- **Domain-Partitioned Routing:** 모든 데이터와 질의는 [커머스, 물류, 무역] 3가지 도메인으로 분류(Categorization)되어 처리됨.
- **Categorization Layer (Gemini Flash-Lite):** 저장 전에는 도메인 분류를, 질의 전에는 의도 파악을 위한 라우팅을 수행.
- **State-Based Overwrite:** 로그 누적 방지, 최종 상태 중심의 데이터 갱신.
- **Trunk & Branch:** 메인 데이터는 LanceDB에 도메인별 필터링 구조로 저장, 이벤트 작업은 임시 서브 채널에서 처리 후 Merge.
- **Task Queue (Orchestration):** 동시 다발적 요청의 순차적 처리를 위한 비동기 작업 큐 도입.

## 3. 핵심 데이터 플로우
### 3.1. 등록/업데이트 파이프라인 (Registration)
1. HTML 수집 -> YAML 정규화
2. **분류(Categorization):** Gemini Flash-Lite로 도메인(커머스/물류/무역) 결정.
3. **LanceDB 저장:** 도메인 메타데이터와 함께 필터링 가능한 테이블에 저장/갱신(URL Hash ID 기반).

### 3.2. 검색/질의 파이프라인 (Retrieval)
1. **의도 라우팅(Categorization):** 사용자의 질의가 어느 도메인인지 분류.
2. **필터링 검색:** 해당 도메인 메타데이터로 LanceDB에서 필터링 후 Full-Text Search 및 벡터 검색 수행.
3. **응답 생성(Sub-track):** 검색 결과와 질의를 Gemini Flash-Lite에 전달하여 최종 답변 생성.

## 4. 데이터 구조 (Schema)
YAML 기반의 계층형 장부 구조에 도메인 태그 추가.

```yaml
domain: "COMMERCE" # 필터링용 메타데이터
data:
  주문번호: ...
  ...
```

## 5. 단계별 기획안
- **Phase 1:** 표준 장부 포맷(YAML) 및 도메인 구조 정의.
- **Phase 2:** HTML 정제 및 정규화 하네스 구현.
- **Phase 3:** 분류(Categorization) 로직 및 LanceDB 하이브리드 검색 구현.
- **Phase 4:** 비동기 작업 큐 및 채널 브랜칭/Merge 파이프라인 구현.
