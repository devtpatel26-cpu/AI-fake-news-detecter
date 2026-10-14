package com.app.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import java.util.Date;

// This annotation defines the name of the collection inside MongoDB
@Document(collection = "scanned_articles")
public class Article {

    @Id
    private String id;               // Auto-generated unique MongoDB Object ID
    private String content;          // The actual text of the news article pasted by user
    private String prediction;       // The result from AI ("REAL" or "FAKE")
    private double fakeProbability;  // Percentage score calculated by Python Deep Learning model
    private Date timestamp;          // Date and time the article was checked

    // Default Constructor (Required by Spring Data MongoDB)
    public Article() {
        this.timestamp = new Date();
    }

    // Parametrized Constructor for easy creation
    public Article(String content, String prediction, double fakeProbability) {
        this.content = content;
        this.prediction = prediction;
        this.fakeProbability = fakeProbability;
        this.timestamp = new Date();
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }

    public double getFakeProbability() {
        return fakeProbability;
    }

    public void setFakeProbability(double fakeProbability) {
        this.fakeProbability = fakeProbability;
    }

    public Date getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Date timestamp) {
        this.timestamp = timestamp;
    }
}
