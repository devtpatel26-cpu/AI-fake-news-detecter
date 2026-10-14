package com.app;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        // This line kicks off the embedded Tomcat web server 
        // and initializes the Spring context configuration.
        SpringApplication.run(Application.class, args);
        System.out.println("🚀 Java Backend Server Core Framework is successfully initialized!");
    }
}
