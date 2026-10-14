/**
 * Triggers the background network pipeline to analyze user text input fields.
 * Sends data packets directly down to the Java backend controller endpoints.
 */
function analyzeText() {
    const textInputField = document.getElementById("newsText");
    const executionButton = document.getElementById("analyzeBtn");
    const resultCardContainer = document.getElementById("resultCard");
    const statusTextBadge = document.getElementById("statusBadge");
    const progressBarFillElement = document.getElementById("probabilityFill");
    const evaluationMetricText = document.getElementById("probabilityText");
    const historyLogsTableBody = document.getElementById("historyTableBody");
    const fallbackNoHistoryRow = document.getElementById("noHistoryRow");

    const textPayload = textInputField.value.trim();

    // Prevent submission if input parameters are completely empty strings
    if (!textPayload) {
        alert("Please paste some text content before initiating an AI evaluation request.");
        return;
    }

    // Set UI state to loading indicators while network operations resolve
    executionButton.disabled = true;
    executionButton.innerText = "Analyzing Content Matrix...";

    // Fire asynchronous payload data straight to the primary Spring Boot API channel
    fetch("/api/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: textPayload })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Server infrastructure returned an operational error.");
        }
        return response.json();
    })
    .then(savedArticleDocument => {
        // 1. Unhide results evaluation summary panel container block
        resultCardContainer.classList.remove("hidden");

        // 2. Adjust classification assessment text badges with explicit state classes
        statusTextBadge.innerText = savedArticleDocument.prediction;
        statusTextBadge.className = "badge " + savedArticleDocument.prediction;

        // 3. Scale and update visual colorized deep learning probability bars
        const calculatedPercentage = savedArticleDocument.fakeProbability;
        progressBarFillElement.style.width = calculatedPercentage + "%";
        evaluationMetricText.innerText = "Fake News Probability: " + calculatedPercentage + "%";

        // Handle color accents depending on risk evaluation ranges
        if (savedArticleDocument.prediction === "FAKE") {
            progressBarFillElement.style.backgroundColor = "#ef4444"; // Red for danger/fake
        } else {
            progressBarFillElement.style.backgroundColor = "#22c55e"; // Green for safe/real
        }

        // 4. Prepend the transaction event record log entry directly onto history tables
        if (fallbackNoHistoryRow) {
            fallbackNoHistoryRow.remove();
        }

        const standardRowTemplate = document.createElement("tr");
        standardRowTemplate.innerHTML = `
            <td class="text-snippet">${escapeHtmlStrings(savedArticleDocument.content)}</td>
            <td><span class="table-badge ${savedArticleDocument.prediction}">${savedArticleDocument.prediction}</span></td>
            <td>${savedArticleDocument.fakeProbability}%</td>
        `;

        historyLogsTableBody.insertBefore(standardRowTemplate, historyLogsTableBody.firstChild);

        // Reset input area fields following safe transactional completions
        textInputField.value = "";
    })
    .catch(networkPipelineException => {
        console.error("Pipeline failure:", networkPipelineException);
        alert("Execution Error: Unable to establish communications with backend processing services.");
    })
    .finally(() => {
        // Restore buttons back to clickable interactive default statuses
        executionButton.disabled = false;
        executionButton.innerText = "Analyze Article";
    });
}

/**
 * Utility helper layer to santize strings and block inner HTML injection attempts.
 */
function escapeHtmlStrings(textInputString) {
    const rawMapElements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return textInputString.replace(/[&<>"']/g, function(characterKey) { 
        return rawMapElements[characterKey]; 
    });
}
