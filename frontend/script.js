document.addEventListener('DOMContentLoaded', () => {
    const analyzeButton = document.getElementById('analyzeButton');
    const userInputField = document.getElementById('userInput');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultCard = document.getElementById('resultCard');
    const resultDisplay = document.getElementById('resultDisplay');
    const errorCard = document.getElementById('errorCard');
    const errorDisplay = document.getElementById('errorDisplay');

    analyzeButton.addEventListener('click', async () => {
        const userInput = userInputField.value.trim();

        if (!userInput) {
            alert('법률 상담 내용을 입력해주세요.');
            return;
        }

        // UI 초기화 및 로딩 상태 표시
        resultCard.classList.add('d-none');
        errorCard.classList.add('d-none');
        loadingSpinner.classList.remove('d-none');
        analyzeButton.disabled = true;

        try {
            const response = await fetch('http://localhost:8080/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'text/plain' // 텍스트로 전송
                },
                body: userInput
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
            }

            const data = await response.json();

            // 결과 표시
            resultDisplay.innerHTML = `
                <p><strong>최종 답변:</strong></p>
                <pre>${data.final_answer}</pre>
                <p><strong>실행 시간:</strong> ${data.execution_time}</p>
            `;
            resultCard.classList.remove('d-none');

        } catch (error) {
            console.error('분석 요청 중 오류 발생:', error);
            errorDisplay.textContent = `오류: ${error.message}. 서버 로그를 확인해주세요.`;
            errorCard.classList.remove('d-none');
        } finally {
            loadingSpinner.classList.add('d-none');
            analyzeButton.disabled = false;
        }
    });
});
