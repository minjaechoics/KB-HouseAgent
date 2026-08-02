# Android/iOS 앱 빌드·배포 가이드

작성 기준일: 2026-07-17  
대상 프로젝트: `jeonse_helper`

## 1. 결론과 권장 구조

이 프로젝트는 Python 에이전트와 데이터베이스를 휴대폰 안에 넣는 방식이 아니라, 현재 FastAPI 서버를 백엔드로 운영하고 Android/iOS 앱은 그 서버를 호출하는 클라이언트로 만드는 것이 적합하다.

```text
Android 앱 ─┐
            ├─ HTTPS ─ FastAPI ─ Agentic LLM ─ OpenAI API
iOS 앱 ────┘                 ├─ 부동산 DB
                             ├─ 금융서비스 DB
                             └─ 지도·외부 데이터 API
```

권장 모바일 프레임워크는 **Flutter**다. 한 코드베이스에서 Android와 iOS 네이티브 앱을 만들 수 있고 현재 REST API를 그대로 호출할 수 있다. 기존 `/gui` 페이지를 WebView로 감싸는 방식은 내부 시연용으로는 빠르지만, 사용자 경험이 제한적이고 Apple의 최소 기능성 심사 기준상 단순 웹사이트 포장으로 판단될 가능성이 있으므로 정식 배포 방식으로 권장하지 않는다.

중요한 보안 원칙은 다음과 같다.

- OpenAI 키, 공공데이터 키, CODEF 키와 비밀키는 앱에 절대 넣지 않는다.
- 앱에는 공개되어도 되는 백엔드 API 주소만 넣는다.
- 현재 소스에 하드코딩된 비밀키도 공개 배포 전에는 서버 환경변수 또는 Secret Manager로 옮기고 기존 키를 폐기·재발급한다.
- 부동산·금융 DB와 LLM 호출은 모두 서버에서 수행한다.

## 2. 현재 API와 모바일 화면의 대응

현재 서버 구현은 `src/server/app.py`에 있으며 모바일 앱은 다음 API를 사용한다.

| 순서 | API | 요청 | 용도 |
|---:|---|---|---|
| 1 | `GET /health` | 없음 | 서버·LLM 상태 확인 |
| 2 | `POST /session` | 사용자 프로필 | 상담 세션 생성 |
| 3 | `POST /chat` | `session_id`, `text` | Agentic RAG 상담 진행 |
| 선택 | `POST /fraud/score` | 매물 속성 객체 | 단일 매물 위험도 계산 |

세션 생성 요청 예시는 다음과 같다.

```json
{
  "age": 29,
  "monthly_income_manwon": 300,
  "total_asset_manwon": 6000,
  "monthly_living_cost_manwon": 120,
  "income_decile": 5,
  "preferred_sido": "대전"
}
```

응답의 `session_id`를 저장한 뒤 대화 요청에 계속 포함한다.

```json
{
  "session_id": "서버에서_받은_값",
  "text": "유성구에서 전세대출을 활용해 살 수 있는 집을 추천해줘"
}
```

앱 화면은 다음 정도로 나누는 것이 적절하다.

1. 프로필 입력: 나이, 월소득, 자산, 생활비, 소득분위, 선호 지역
2. 상담 화면: 사용자 메시지, AI 종합 답변, 추가 질문과 확인 단계
3. 결과 카드: 추천 매물, 적용 가능한 금융상품, 월 부담액과 자금계획, 근거 URL
4. 매물 상세: 가격·면적·거래유형·지역·위험도·금융 조합
5. 설정: 개인정보 처리방침, 데이터 삭제 요청, 문의처, 면책 안내

`agent_trace`와 SQL/RAG 디버그 정보는 개발 빌드에서만 표시하고 운영 앱에서는 숨긴다. API 키, 시스템 프롬프트, 내부 예외 스택은 어느 빌드에서도 노출하면 안 된다.

## 3. 앱을 만들기 전에 서버에서 고칠 항목

현재 API는 데모에는 쓸 수 있지만 그대로 다중 사용자 앱을 배포하기에는 부족하다.

### 3.1 필수 수정

- **HTTPS**: `https://api.example.com`처럼 인증서가 적용된 주소를 사용한다.
- **세션 공유 저장소**: 현재 `_SESSIONS`는 프로세스 메모리다. 서버 재시작 시 사라지고 여러 worker 사이에서 공유되지 않는다. Redis 또는 영속 DB로 바꾼다.
- **인증과 세션 만료**: 최소한 익명 기기 토큰 또는 로그인 토큰, 세션 TTL, 세션 소유자 검사를 둔다.
- **API 버전**: `/v1/session`, `/v1/chat`처럼 버전을 고정한다.
- **요청·응답 스키마**: `/chat`의 성공·추가질문·확인·결과·오류 응답을 Pydantic 모델로 고정한다.
- **오류 은닉**: 현재 500 응답에 Python 예외 메시지가 포함된다. 앱에는 오류 코드와 안전한 사용자 메시지만 반환하고 상세 오류는 서버 로그에만 남긴다.
- **요청 제한**: IP·사용자별 rate limit, 입력 길이 제한, LLM 비용 상한을 둔다.
- **관측성**: request ID, 구조화 로그, LLM 지연·실패율·토큰 비용, DB 쿼리 시간, 알림을 설정한다.
- **개인정보 수명주기**: 보관 기간, 삭제 API, 로그 마스킹, 백업 삭제 정책을 정의한다.
- **중복 요청 방지**: 네트워크 재시도로 같은 `/chat`이 두 번 처리되지 않도록 idempotency key를 지원한다.

Redis 전환 전에는 운영 명령도 worker를 하나만 사용해야 한다.

```cmd
py -3 -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

현재 파일 주석에 있는 4-worker 예시는 인메모리 세션 상태에서는 사용하면 안 된다. Redis 전환 후에만 여러 worker를 사용한다.

### 3.2 CORS에 대한 구분

Android/iOS 네이티브 앱의 HTTP 클라이언트는 브라우저가 아니므로 CORS 설정이 필요하지 않다. 향후 Flutter Web이나 웹 GUI를 다른 도메인에서 제공할 때만 FastAPI `CORSMiddleware`에 허용할 정확한 origin을 등록한다. 운영에서 `*`와 credential 허용을 함께 사용하지 않는다.

## 4. 개발 환경 준비

### 4.1 공통

1. [Flutter 설치 안내](https://docs.flutter.dev/install)에 따라 stable Flutter SDK를 설치한다.
2. Git과 IDE(Android Studio 또는 VS Code)를 설치한다.
3. 다음 명령으로 누락된 도구를 확인한다.

```cmd
flutter doctor -v
```

### 4.2 Android

- Windows 또는 macOS에서 개발·서명·빌드할 수 있다.
- Android Studio, Android SDK, platform tools, emulator를 설치한다.
- `flutter doctor --android-licenses`로 SDK 라이선스를 확인한다.

### 4.3 iOS

- iOS 시뮬레이터, 실기기 서명, IPA 생성은 **macOS와 Xcode가 필요**하다. 현재 Windows PC만으로 iOS 정식 빌드를 만들 수 없다.
- App Store 배포에는 Apple Developer Program 계정이 필요하다.
- 2026-04-28부터 App Store Connect 업로드는 Xcode 26 이상과 iOS 26 SDK 이상을 요구하므로, 빌드 시점의 [Apple 제출 요구사항](https://developer.apple.com/news/upcoming-requirements/)을 다시 확인한다.
- CocoaPods를 설치하고 Xcode 초기 설정과 라이선스 동의를 마친다.

Windows에서 대부분의 Flutter/Dart 코드를 개발하고 Git으로 Mac에 넘겨 iOS만 빌드해도 된다. 단, iOS 화면·권한·서명 문제는 반드시 Mac과 실제 iPhone에서도 테스트한다.

## 5. Flutter 프로젝트 생성

기존 Python 프로젝트와 모바일 프로젝트를 형제 디렉터리로 두는 구성을 권장한다.

```cmd
cd /d D:\연구\FlexML\kb
flutter create --org kr.co.flexml --platforms=android,ios jeonse_helper_mobile
cd jeonse_helper_mobile
flutter pub add http
flutter pub add flutter_secure_storage
```

예상 구조는 다음과 같다.

```text
kb/
├─ jeonse_helper/                 # 현재 Python 백엔드
└─ jeonse_helper_mobile/          # 새 Flutter 앱
   ├─ android/
   ├─ ios/
   ├─ lib/
   │  ├─ main.dart
   │  ├─ config/app_config.dart
   │  ├─ api/api_client.dart
   │  ├─ models/
   │  ├─ screens/
   │  │  ├─ profile_screen.dart
   │  │  └─ chat_screen.dart
   │  └─ widgets/
   ├─ test/
   └─ pubspec.yaml
```

Android application ID와 iOS bundle ID는 예를 들어 `kr.co.flexml.jeonsehelper`처럼 소유한 역도메인으로 정한다. 스토어 등록 후 바꾸기 어려우므로 처음에 확정한다.

## 6. API 주소와 환경 분리

API 주소는 소스에 하나로 고정하지 말고 `--dart-define`으로 개발·스테이징·운영을 분리한다. 이것은 비밀 저장이 아니라 빌드별 주소 선택을 위한 것이다.

`lib/config/app_config.dart` 예시:

```dart
class AppConfig {
  static const apiBaseUrl = String.fromEnvironment('API_BASE_URL');

  static void validate() {
    if (apiBaseUrl.isEmpty) {
      throw StateError('API_BASE_URL is required');
    }
  }
}
```

환경별 예시는 다음과 같다.

| 환경 | API 주소 예 |
|---|---|
| Android Emulator | `http://10.0.2.2:8000` |
| iOS Simulator | `http://127.0.0.1:8000` |
| 같은 Wi-Fi의 실제 폰 | `http://개발PC_LAN_IP:8000` |
| 스테이징 | `https://staging-api.example.com` |
| 운영 | `https://api.example.com` |

로컬 HTTP 예외를 운영 설정에 넣지 않는 것이 중요하다. Android 9 이상은 기본적으로 cleartext HTTP를 제한하고 iOS도 App Transport Security가 안전하지 않은 연결을 제한한다. 기기 테스트도 가능하면 HTTPS 개발 터널이나 개발 인증서가 적용된 스테이징 서버를 사용한다. 꼭 로컬 HTTP가 필요하면 개발 빌드에만 호스트 한정 예외를 만들고 release manifest/Info.plist에서는 제거한다.

관련 공식 문서: [Android Network Security Configuration](https://developer.android.com/privacy-and-security/security-config), [Apple의 안전하지 않은 네트워크 연결 방지](https://developer.apple.com/documentation/security/preventing-insecure-network-connections)

## 7. Flutter에서 현재 API 호출하기

아래 코드는 최소 연동 예시다. 실제 앱에서는 오류 모델, 로깅, 취소, 인증 토큰, idempotency key를 추가한다.

```dart
import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';

class ApiException implements Exception {
  final int statusCode;
  final String message;
  ApiException(this.statusCode, this.message);
}

class ApiClient {
  final http.Client client;
  ApiClient({http.Client? client}) : client = client ?? http.Client();

  Uri _uri(String path) => Uri.parse('${AppConfig.apiBaseUrl}$path');

  Future<String> createSession(Map<String, dynamic> profile) async {
    final response = await client
        .post(
          _uri('/session'),
          headers: {'Content-Type': 'application/json; charset=utf-8'},
          body: jsonEncode(profile),
        )
        .timeout(const Duration(seconds: 20));

    final body = jsonDecode(utf8.decode(response.bodyBytes));
    if (response.statusCode != 200) {
      throw ApiException(response.statusCode, '세션을 만들 수 없습니다.');
    }
    return body['session_id'] as String;
  }

  Future<Map<String, dynamic>> chat(String sessionId, String text) async {
    final response = await client
        .post(
          _uri('/chat'),
          headers: {'Content-Type': 'application/json; charset=utf-8'},
          body: jsonEncode({'session_id': sessionId, 'text': text}),
        )
        .timeout(const Duration(seconds: 120));

    final decoded = jsonDecode(utf8.decode(response.bodyBytes));
    if (response.statusCode != 200) {
      throw ApiException(response.statusCode, '상담 응답을 받지 못했습니다.');
    }
    return decoded as Map<String, dynamic>;
  }
}
```

Flutter의 공식 JSON POST 예제는 [Send data to the internet](https://docs.flutter.dev/cookbook/networking/send-data)를 참고한다.

LLM 응답은 일반 API보다 오래 걸릴 수 있으므로 로딩 상태와 취소 UI를 제공한다. 시간 초과 후 `/chat`을 무조건 재전송하면 같은 대화가 중복 실행될 수 있다. 서버에 idempotency 처리를 넣기 전에는 사용자에게 재시도 여부를 알리고, 세션 생성이나 health check처럼 안전한 작업만 자동 재시도한다.

`session_id`는 `flutter_secure_storage`에 보관할 수 있지만 앱의 사용자 인증 토큰을 대신하지는 않는다. 서버가 재시작되어 404가 오면 현재 구조에서는 새 세션을 만든 뒤 프로필과 필요한 대화 상태를 복구해야 한다. Redis 전환 후에는 명시적인 만료 응답 코드를 정의한다.

## 8. 로컬 연결 테스트

### 8.1 백엔드 실행

Windows CMD 한 줄:

```cmd
cd /d D:\연구\FlexML\kb\jeonse_helper && set JEONSE_LLM=api && py -3 -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

먼저 브라우저에서 `http://127.0.0.1:8000/health`가 정상인지 확인한다. 실제 폰에서 접속할 때는 Windows 방화벽에서 포트 8000의 사설 네트워크 인바운드 접근이 필요할 수 있으며, PC와 폰이 같은 Wi-Fi에 있어야 한다.

### 8.2 Android Emulator 실행

```cmd
cd /d D:\연구\FlexML\kb\jeonse_helper_mobile
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000
```

Android 앱에는 인터넷 권한이 필요하다. `android/app/src/main/AndroidManifest.xml`의 `<manifest>` 바로 아래에 확인한다.

```xml
<uses-permission android:name="android.permission.INTERNET" />
```

### 8.3 iOS Simulator 실행

Mac 터미널에서:

```bash
cd /path/to/jeonse_helper_mobile
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

백엔드가 Windows PC에서 실행 중이라면 `127.0.0.1` 대신 그 PC의 LAN IP 또는 HTTPS 스테이징 주소를 사용한다.

## 9. 응답을 앱 UI로 표현하는 규칙

현재 에이전트 응답은 대화 상태에 따라 형태가 달라질 수 있으므로 단순히 모든 결과를 같은 목록으로 표시하면 안 된다.

- 추가 정보가 필요하면 질문 화면으로 표시한다.
- 사용자의 최종 확인이 필요하면 확인/수정 버튼을 표시한다.
- 종합 추천이면 LLM의 설명을 먼저 보여주고 매물과 금융상품을 근거 카드로 분리한다.
- 결과가 없으면 적용된 조건과 완화할 수 있는 조건을 보여준다.
- 금융상품은 이름, 지역, 금리, 한도, 신청기간, 공식 출처를 표시한다.
- 매물은 거래유형, 주택유형, 주소, 보증금/월세/매매가, 면적, 위험도와 합성 데이터 여부를 표시한다.
- 개발 모드에서만 `agent_trace`, SQL, tool call을 접을 수 있는 패널로 표시한다.

API 스키마가 안정되면 FastAPI의 `/openapi.json`을 기준으로 Dart 모델을 생성하거나 직접 typed model을 만든다. `Map<String, dynamic>`은 최초 연결 확인까지만 사용하는 편이 안전하다.

## 10. Android 정식 빌드와 Google Play 배포

### 10.1 앱 정보와 SDK 확인

1. 앱 이름, application ID, 아이콘, 스플래시, 버전을 정한다.
2. `pubspec.yaml`의 버전을 `1.0.0+1` 형식으로 설정한다.
3. Flutter/Android Gradle Plugin이 지원하는 최신 compile/target SDK를 사용한다.
4. 2026-07-17 현재 신규 앱과 업데이트 제출은 Android 15(API 35) 이상이어야 하고, **2026-08-31부터 Android 16(API 36) 이상**이 요구된다. 출시일이 가깝기 때문에 API 36으로 준비하고 제출 직전 [Google Play 대상 API 요구사항](https://support.google.com/googleplay/android-developer/answer/11926878?hl=en)을 다시 확인한다.

### 10.2 업로드 키 생성

Windows 예시:

```cmd
keytool -genkeypair -v -keystore "%USERPROFILE%\upload-keystore.jks" -keyalg RSA -keysize 2048 -validity 10000 -alias upload
```

Flutter 공식 [Android 앱 빌드·출시 안내](https://docs.flutter.dev/deployment/android)에 따라 `android/key.properties`와 Gradle signing config를 연결한다.

- keystore, 비밀번호, `key.properties`를 Git에 커밋하지 않는다.
- 별도의 암호화 백업을 만들고 접근자를 제한한다.
- Google Play App Signing을 사용하더라도 업로드 키 관리는 필요하다.

### 10.3 검사와 AAB 생성

```cmd
flutter clean
flutter pub get
flutter analyze
flutter test
flutter build appbundle --release --dart-define=API_BASE_URL=https://api.example.com
```

기본 출력 위치:

```text
build/app/outputs/bundle/release/app-release.aab
```

Google Play 신규 앱은 게시 형식으로 Android App Bundle(AAB)을 사용한다. 자세한 개념은 [Android App Bundle 공식 안내](https://developer.android.com/guide/app-bundle)를 참고한다.

### 10.4 Play Console 순서

1. Google Play Console에서 앱을 만든다.
2. 스토어 설명, 아이콘, feature graphic, 휴대폰 스크린샷, 지원 연락처를 등록한다.
3. 개인정보처리방침 URL과 Data safety 양식을 작성한다.
4. 콘텐츠 등급, 대상 연령, 광고 여부, 앱 접근 방법을 입력한다.
5. AAB를 Internal testing 트랙에 먼저 올린다.
6. 실제 사용자 계정/프로필로 Agentic RAG, 서버 오류, 느린 네트워크를 검증한다.
7. Closed testing 또는 Open testing을 거친 뒤 Production 심사를 요청한다.

Data safety에는 앱 자체뿐 아니라 OpenAI 등 서버 측 제3자 처리와 포함한 SDK의 데이터 수집·공유도 실제 동작에 맞춰 작성한다. 최신 양식은 [Google Play Data safety 안내](https://support.google.com/googleplay/android-developer/answer/10787469?hl=en)를 기준으로 한다.

## 11. iOS 정식 빌드와 App Store 배포

### 11.1 Apple 등록과 Xcode 설정

1. Apple Developer 계정에서 고유 bundle ID를 등록한다.
2. App Store Connect에서 같은 bundle ID로 앱 레코드를 만든다.
3. Mac에서 `ios/Runner.xcworkspace`를 연다.
4. Runner target의 Team, Bundle Identifier, Signing을 설정한다.
5. 자동 서명을 사용하거나 배포 인증서와 provisioning profile을 연결한다.
6. `pubspec.yaml`의 버전과 build number를 설정한다. App Store Connect에 올리는 각 build number는 고유해야 한다.

### 11.2 검사와 IPA 생성

```bash
flutter clean
flutter pub get
flutter analyze
flutter test
flutter build ipa --release --dart-define=API_BASE_URL=https://api.example.com
```

기본 출력 위치:

```text
build/ios/archive/Runner.xcarchive
build/ios/ipa/*.ipa
```

공식 절차와 출력 위치는 [Flutter iOS 앱 빌드·출시 안내](https://docs.flutter.dev/deployment/ios)를 따른다.

### 11.3 TestFlight와 심사

1. Xcode Organizer 또는 Apple Transporter로 IPA를 App Store Connect에 업로드한다.
2. 암호화 사용 여부, 수출 규정, 개인정보 항목을 입력한다.
3. Internal TestFlight로 팀 테스트를 한다.
4. External TestFlight가 필요하면 베타 심사를 진행한다.
5. 스크린샷, 설명, 지원 URL, 개인정보처리방침 URL, 심사 메모를 작성한다.
6. 심사자가 기능을 재현할 수 있도록 테스트 계정 또는 로그인 없는 경로를 제공한다.
7. 백엔드 스테이징/운영 서버가 심사 기간 내내 접근 가능해야 한다.
8. App Review를 제출하고 승인 후 수동 또는 자동 출시한다.

전체 흐름은 [App Store Connect workflow](https://developer.apple.com/help/app-store-connect/get-started/app-store-connect-workflow/)를 참고한다.

## 12. 개인정보·금융·AI 안내

이 앱은 나이, 소득, 자산, 생활비, 선호 지역, 자유형식 대화처럼 민감할 수 있는 정보를 처리한다. 개인정보처리방침에는 최소한 다음 내용을 실제 구현과 일치하게 적는다.

- 수집하는 각 데이터와 수집 목적
- 서버 및 로그 보관 기간
- OpenAI 등 제3자 AI 제공자에게 전달되는 데이터 범위와 처리 목적
- 금융·부동산 DB와 외부 API 제공자
- 암호화, 접근통제, 삭제 요청 방법과 문의처
- 계정을 만들 경우 계정·관련 데이터 삭제 방법
- 분석, 충돌 보고, 푸시 SDK 등 제3자 SDK의 수집 항목

Apple은 새 앱과 업데이트 제출 시 앱과 제3자 파트너의 데이터 처리 내용을 App Privacy에 공개하도록 요구한다. [Apple App Privacy Details](https://developer.apple.com/app-store/app-privacy-details/)와 [App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)를 실제 구현과 대조한다. 특히 사용자 데이터를 제3자 AI로 보내는 사실과 목적을 사용 전에 명확하게 알리고 필요한 동의를 받도록 설계한다.

서비스 표현도 다음처럼 구분한다.

- 이 서비스는 금융상품과 주거 선택을 돕는 정보·추천 서비스이며 대출 실행 기관이 아님을 표시한다.
- 실제 자격, 금리, 한도와 신청 가능 여부는 금융기관·공공기관의 최종 심사와 최신 공고를 따른다고 표시한다.
- 생성한 부동산 데이터는 **실제 판매 중인 매물이 아니라 합성 데이터**임을 결과마다 명확히 표시한다.
- 실제 매물 서비스를 추가할 때는 중개사·제휴처 출처, 갱신 시각, 거래 가능 여부를 분리해 표시한다.
- LLM 답변 자체와 DB/공식 출처에서 조회한 사실을 UI에서 구분한다.

향후 앱 안에서 실제 대출 중개·신청·계좌 연결까지 수행한다면 금융 규제와 스토어의 금융서비스 추가 요건이 달라진다. 그 단계에서는 출시 국가의 법률 검토와 사업자 자격 확인을 별도로 진행해야 한다.

## 13. 출시 전 테스트 체크리스트

### 기능

- 프로필 생성 후 `session_id`가 정상 저장되는가
- 자연어 질문이 `/chat`에 한 번만 전송되는가
- 추가질문, 사용자 확인, 추천, 결과 없음 상태가 각각 다르게 표시되는가
- 매물 DB와 금융 DB 근거가 종합 답변과 일치하는가
- 한국어, 긴 입력, 특수문자, 빈 입력이 안전하게 처리되는가
- 서버 재시작·세션 만료 시 프로필 복구 또는 새 세션 안내가 되는가
- 합성 매물 표시와 금융정보 면책이 보이는가

### 네트워크와 장애

- 비행기 모드, 느린 3G 수준, DNS 실패, 5xx, timeout을 처리하는가
- 로딩 중 중복 탭이 막히는가
- 실패한 대화의 무조건 자동 재전송을 피하는가
- 앱을 백그라운드로 보냈다가 돌아와도 상태가 일관적인가
- 서버가 로그에 API 키·전체 개인정보·사용자 원문을 불필요하게 남기지 않는가

### 보안

- APK/AAB/IPA와 Dart define 안에 OpenAI·CODEF·공공데이터 비밀키가 없는가
- release 앱은 HTTPS만 사용하는가
- 디버그 trace, SQL, stack trace가 운영 UI와 로그에 노출되지 않는가
- 세션 소유권, 만료, rate limit, 입력 크기 제한이 적용되는가
- 개인정보 삭제 요청을 실제로 이행할 수 있는가

### 품질과 스토어

- Android 저사양 기기와 여러 화면 크기에서 테스트했는가
- iPhone 실제 기기와 최신/지원 최소 iOS에서 테스트했는가
- Play Data safety와 Apple App Privacy 답변이 실제 네트워크 동작과 일치하는가
- 심사 계정, 지원 URL, 개인정보처리방침 URL이 외부에서 열리는가
- API 서버와 공식 출처 링크가 심사 기간 동안 정상인가

## 14. 권장 구현 순서

1. `/session`과 `/chat` 응답 스키마를 고정하고 API를 `/v1`으로 만든다.
2. Redis 세션, 인증, 오류 코드, rate limit, idempotency를 서버에 적용한다.
3. Flutter 프로필 화면과 API client를 만들고 세션 생성을 연결한다.
4. 채팅과 에이전트 상태별 UI를 만든다.
5. 매물·금융 근거 카드와 자금계획 화면을 만든다.
6. HTTPS 스테이징 서버에서 Android/iOS 실기기 통합 테스트를 한다.
7. 개인정보처리방침, 합성 데이터 표시, 금융 면책과 삭제 절차를 완성한다.
8. Android internal testing과 iOS TestFlight를 운영한다.
9. 오류율·LLM 비용·추천 품질을 확인한 뒤 스토어 심사를 제출한다.

## 15. 최소 완료 기준

다음 조건이 모두 충족되면 “앱으로 빌드 가능” 단계를 넘어 “스토어 테스트 제출 가능” 상태로 볼 수 있다.

- Android와 iOS가 같은 운영 HTTPS API로 정상 상담한다.
- 모바일 바이너리에 외부 서비스 비밀키가 없다.
- 서버 재시작과 다중 worker에서도 세션이 유지된다.
- 에이전트 응답 상태별 typed response와 UI가 있다.
- 실제 LLM 호출 실패, DB 조회 실패, timeout의 사용자 안내와 서버 fallback이 검증됐다.
- 합성 부동산 데이터와 실제 정책 정보가 명확히 구분된다.
- 개인정보 삭제, 앱 개인정보 공개, 스토어 심사용 자료가 준비됐다.
- Android AAB와 iOS IPA가 release 설정으로 생성되고 내부 배포에서 검증됐다.

