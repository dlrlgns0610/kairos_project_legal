package com.kairos.legal_analysis_backend;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api")
public class LegalAnalysisController {

    @CrossOrigin(origins = "*") // CORS 허용
    @PostMapping("/analyze")
    public ResponseEntity<String> analyzeLegalCase(@RequestBody String userInput) {
        try {
            // Python 스크립트 실행 명령
            // python3 대신 python을 사용하거나, 환경에 맞는 경로를 지정해야 할 수 있습니다.
            ProcessBuilder pb = new ProcessBuilder("python3", "-m", "legal_multiagent.main");
            pb.directory(new java.io.File("..")); // 프로젝트 루트 디렉토리에서 실행

            Process p = pb.start();

            // Python 스크립트에 입력 전달
            try (OutputStream os = p.getOutputStream()) {
                os.write(userInput.getBytes("UTF-8"));
                os.flush();
            }

            // Python 스크립트의 표준 출력 읽기
            String pythonOutput = new BufferedReader(
                new InputStreamReader(p.getInputStream(), "UTF-8"))
                .lines().collect(Collectors.joining("\n"));

            // Python 스크립트의 표준 에러 읽기 (디버깅용)
            String pythonError = new BufferedReader(
                new InputStreamReader(p.getErrorStream(), "UTF-8"))
                .lines().collect(Collectors.joining("\n"));

            int exitCode = p.waitFor();

            if (exitCode != 0) {
                System.err.println("Python script exited with error code: " + exitCode);
                System.err.println("Python error output: " + pythonError);
                return ResponseEntity.status(500).body("{\"error\": \"Python analysis failed\", \"details\": \"" + pythonError.replace("\"", "\\\"") + "\"}");
            }

            return ResponseEntity.ok(pythonOutput);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("{\"error\": \"Server internal error\", \"details\": \"" + e.getMessage().replace("\"", "\\\"") + "\"}");
        }
    }
}
