## 📦 Windows Release v1.5.0 [Build 260214]

This release includes the standalone executable for Windows. Please follow the setup instructions below to ensure the engine runs correctly.

### **🛠 Installation & Execution (English)**
1. **Download:** Download `MaxEnt_Auto_Engine.zip` and extract it.
2. **Core Engine Setup: You MUST place the maxent.jar file in the following specific directory:
   - **`C:\maxent\maxent.jar`**
   - *Note: If the folder does not exist, you must create it manually.*
3. **Java Dependency:** Ensure **Java (JRE)** is installed on your PC.
4. **Run:** Execute **`MaxEnt_Auto_Engine.exe`**.

### **🛠 설치 및 실행 방법 (한국어)**
1. **다운로드:** `MaxEnt_Auto_Engine.zip` 파일의 압축을 풉니다.
2. **코어 엔진 설정:** **`maxent.jar`** 파일을 반드시 아래 경로로 선택하세요:
   - **`C:\maxent\maxent.jar`**
   - *주의: 폴더가 없다면 C드라이브에 직접 생성해야 합니다.*
3. **Java 설치:** 시스템에 **Java (JRE)**가 설치되어 있어야 합니다.
4. **실행:** **`MaxEnt_Auto_Engine.exe`**를 실행합니다.

---

### **📂 Required Environment**
- **OS:** Windows 10 / 11 (64-bit)
- **Mandatory Path:** `C:\maxent\maxent.jar`


C:\
└── maxent\
    └── maxent.jar                 # Essential Core File

[Your Extraction Folder]\
├── MaxEnt_Auto_Engine.exe        # Executable Application
├── maxent_Engine_config.ini       # Auto-generated Configuration
├── 01.SpeicesLIST/                       # [Input] Species occurrence data (.csv)
├── 02.ASCII_DATA/                       # [Input] Environmental ASCII layers (.asc)
└── 03.Results/                              # [Output] Auto-generated Results


Note for Users: The engine is pre-configured to look for the core logic at the fixed path C:\maxent\maxent.jar. Please ensure this setup is completed before running the executable to prevent initialization errors.

**Full Changelog**: https://github.com/you0742/Maxent_Auto_Engine/commits/v1.4.9-R3

**Full Changelog**: https://github.com/you0742/Maxent_Auto_Engine/commits/V1.5