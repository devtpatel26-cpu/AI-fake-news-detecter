package com.app.controller;

import com.app.model.Article;
import com.app.repository.Repo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/")
public class AppCtrl {

    @Autowired
    private Repo articleRepository;

    // Local address of the running Step 1 Python FastAPI Microservice
    private final String AI_ENGINE_URL = "http://127.0.0";

    /**
     * Renders the home screen interface dashboard view.
     * Pulls the history records directly from MongoDB to display dynamically.
     */
    @GetMapping
    public String viewDashboard(Model model) {
        List<Article> history = articleRepository.findAllByOrderByTimestampDesc();
        model.addAttribute("historyList", history);
        return "index"; // Maps directly to index.ejs
    }

    /**
     * API endpoint to handle real-time content checking.
     * Routes queries asynchronously from frontend JavaScript down to the AI layer.
     */
    @PostMapping("/api/analyze")
    @ResponseBody
    public ResponseEntity<?> analyzeArticleText(@RequestBody Map<String, String> requestPayload) {
        String originalText = requestPayload.get("text");

        if (originalText == null || originalText.trim().isEmpty()) {
            Map<String, String> errorResponse = new HashMap<>();
            errorResponse.put("error", "Input content cannot be empty.");
            return ResponseEntity.badRequest().body(errorResponse);
        }

        try {
            // 1. Prepare standard HTTP request packet for the Python Microservice
            RestTemplate restClient = new RestTemplate();
            HttpHeaders networkHeaders = new HttpHeaders();
            networkHeaders.setContentType(MediaType.APPLICATION_JSON);

            Map<String, String> pythonBody = new HashMap<>();
            pythonBody.put("text", originalText);

            HttpEntity<Map<String, String>> requestPacket = new HttpEntity<>(pythonBody, networkHeaders);

            // 2. Transmit content payload downstream to Python FastAPI engine
            ResponseEntity<Map> aiResponse = restClient.postForEntity(AI_ENGINE_URL, requestPacket, Map.class);
            Map<String, Object> aiResultMap = aiResponse.getBody();

            if (aiResultMap == null) {
                throw new RuntimeException("Empty response payload received from deep learning engine.");
            }

            // 3. Extract calculated evaluation scores from deep learning layers
            String classificationResult = (String) aiResultMap.get("prediction");
            double probabilityScore = ((Number) aiResultMap.get("fake_probability")).doubleValue();

            // 4. Instantiated structured Document Object and pipe it directly to MongoDB
            Article documentRecord = new Article(originalText, classificationResult, probabilityScore);
            Article savedDocument = articleRepository.save(documentRecord);

            // 5. Return newly created document model directly back to browser script
            return ResponseEntity.ok(savedDocument);

        } catch (Exception pipelineException) {
            Map<String, String> failureDetails = new HashMap<>();
            failureDetails.put("error", "AI Engine connection failure: " + pipelineException.getMessage());
            return ResponseEntity.status(500).body(failureDetails);
        }
    }
}
