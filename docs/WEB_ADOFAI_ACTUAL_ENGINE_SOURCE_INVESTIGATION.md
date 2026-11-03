# Web ADOFAI 원본 엔진 추적 결과 (2026-11-03)

요청: `https://web-adofai.impl1113.dev/levels`의 **실제 검색·플레이 엔진을 그대로 확보하고**, 별도 인게임 편집 기능과 직접 커스텀 플레이 기능을 하나의 웹앱으로 연결.

## 1. 실제 운영 중인 엔진 파일 확인

아래는 **실제 사이트의 실행 브라우저가 HTTP 200으로 로드한 자바스크립트**다. GitHub의 다른 ADOFAI 플레이어나 우리가 만든 모형 코드가 아니다.

| 역할 | 원본 사이트 파일 | SHA-256 |
| --- | --- | --- |
| 레벨 목록 라우팅 / 로컬 레벨 캐시 / 게임 편집·플레이 UI / replay player | [app-router.component-BF3hr6k2.js](https://web-adofai.impl1113.dev/assets/app-router.component-BF3hr6k2.js) | `45a6647f424b6a660083fb81ed20bdb142a6f47894a647ae4ed4521274e187b1` |
| 핵심 렌더링·게임 컴포넌트 | [button.component-BR8Evgkc.js](https://web-adofai.impl1113.dev/assets/button.component-BR8Evgkc.js) | `4312d45bc13eb1f9c6b3f2ef57d3b598c422a3c2b8bc99a47998a6e5fb8e15fa` |
| Web ADOFAI 차트 iframe 플레이어(뷰어 전용) | [chart-viewer.entry-DkKg0MKe.js](https://web-adofai.impl1113.dev/assets/chart-viewer.entry-DkKg0MKe.js) | `d2d208d5e3dac787c87491760ccd798ea70840f181df19864332b63565972015` |
| ADOFAI 차트 파서 | [vendor-parser-Dz5hYVfq.js](https://web-adofai.impl1113.dev/assets/vendor-parser-Dz5hYVfq.js) | `44b5f9d501f4b2f9c6e2800648038dccb38a7bd80af3e4973ad8262dc031485f` |
| React 등 런타임 | [vendor-o6MECFhW.js](https://web-adofai.impl1113.dev/assets/vendor-o6MECFhW.js) | `dbf599fae658001c0c2deb2e3ea280e97a37980e86a95e132846e1730782bb3a` |
| 레벨 ZIP 압축 해제 | [tuf-archive-extraction.worker-CxaEfFIK.js](https://web-adofai.impl1113.dev/assets/tuf-archive-extraction.worker-CxaEfFIK.js) | `39195edab8653f75808a8d00b783959a864aaaa25fed39f56c40907714931048` |

배포 JS 파일 총 10개 약 5.5 MB를 **클라우드 임시 분석 환경에서 수집하고 해시 검증**했다. 제삼자 저작권을 가진 파일 자체는 이 공개 저장소에 복사하지 않았다.

## 2. 실제 엔진의 경로 및 파일 구조

라우터 실코드(`app-router.component`) 확인:

- `/levels`: 레벨 검색 및 목록
- `/levels/:id`: 원본 게임 화면 (편집·플레이 모드 상태 존재)
- `/levels/local`: 로컬 레벨 라이브러리로 진입
- `/levels/cache/:packageId`: 로컬에 저장한 레벨 패키지를 불러오는 게임 화면
- `/replay`, `/replay/embed`: 별도의 리플레이 재생 화면
- `/embed/chart?parentOrigin=...`: 외부 앱에 허용된 원본 **차트 시각화** iframe

로컬 ZIP을 사이트가 불러올 때의 저장 공간:

- IndexedDB: `adofai-web-editor-level-packages` (version 1)
- object stores: `packages`, `fileBlobs`
- 보조 OPFS 경로: `adofai-web-editor/level-packages`
- `/levels/cache/:packageId`는 원본 도메인의 스토리지에서 해당 ID를 조회한다.

브라우저에서 `/levels`을 열어 **검색 결과 8,428개**와 `Local Levels` 모달의 `Choose folder`를 실측 확인했다.

## 3. 편집 → 게임 직통 연결을 가로막는 구체적인 인터페이스

### A. `/embed/chart` — 시각화만 공개됨

`chart-viewer.entry`의 `web-adofai.chart` 버전 1 메시지 파서는 `chart.load`, `chart.focus`, `chart.clear`, `markers.set`, `run.focus`, `run.clear` 및 `run.fit-all`을 처리한다.

`chart.load`는 `levelText`를 렌더러에 보내지만 **실제 플레이 시작 명령은 없다.** 확인되는 capability는 `marker-navigation-v1`. 이 API를 '원본 커스텀 게임 실행'이라고 부르면 거짓이다.

### B. `/replay/embed` — 별도의 승인된 호스트 전용

실코드 `lx`에서 `protocolVersion=3`, UUID 세션, `parentOrigin`을 검사한다. 실제 프로덕션 번들의 환경 변수 객체가 `rx={}`로 컴파일되어 있어서 허용된 부모 origin은 **`https://tuforums.com` 하나**다. 다른 origin에서는 `player.load` 요청을 처리하지 않는다.

**제3자 도메인에서 임의로 전송하는 `player.load`는 지원된 공개 명령이 아니다.** 이 보안 검사를 우회하지 않았다.

### C. `/levels/cache/:id` — 도메인별 스토리지 분리

원본 게임의 로컬 레벨 캐시는 `web-adofai.impl1113.dev` 원본 도메인에 속한다. 다른 도메인의 웹앱은 브라우저 동일 출처 정책으로 해당 IndexedDB에 직접 차트를 쓸 수 없다.

## 4. 원본 소스의 상태

- 네트워크로 제공되는 **프로덕션 번들 코드는 실제로 확보**했다.
- 원본 `.ts/.tsx` 파일, 수정 가능한 Vite 프로젝트, 의존성 잠금 파일, 원본 Git 저장소는 **공개적으로 확인되지 않았다**.
- `app-router.component-BF3hr6k2.js.map`은 소스맵 대신 SPA HTML을 반환한다. 다른 파서 JS 맵 역시 동일. 배포 JS에 `sourceMappingURL`이 없다.
- 배포 응답은 다른 출처의 모듈 스크립트 실행에 필요한 `Access-Control-Allow-Origin`을 제공하지 않는다.
- GitHub 공개 코드 검색에서 로컬 DB 식별자인 `adofai-web-editor-level-packages`의 원본 저장소를 찾지 못했다.
- 외부 실행 코드의 수정·재호스팅을 허가하는 라이선스도 발견하지 못했다. 별도 허가 없이 제삼자 엔진을 통째로 공개 저장소·배포 서버에 재배포하지 않는다.

## 5. 현재 결과와 남은 정확한 과제

**확인한 실제 기능**: 원본 검색·플레이 사이트 사용, 원본 차트 뷰어와 통신, 원본 레벨 캐시 로딩 및 로컬 레벨 폴더 가져오기.

**아직 미완성**: 원본 게임 엔진을 사용자 사이트의 독립 런타임으로 이전, 편집 즉시 원본 게임 플레이 시작, 원본 엔진과 통합된 타일·이벤트 UI. 라이선스 없는 전체 JS 복사를 완성 포트라고 주장하지 않는다.

이 문서는 자체 제작 플레이어 또는 '비슷한 사이트' 추천이 아니라, **실제 요청한 첫 번째 사이트의 구동 코드와 제약을 조사한 결과**다.
