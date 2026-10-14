package com.app.repository;

import com.app.model.Article;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface Repo extends MongoRepository<Article, String> {
    
    // Automatically generates a query to fetch the latest scans ordered by date
    List<Article> findAllByOrderByTimestampDesc();
}
