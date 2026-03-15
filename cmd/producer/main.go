package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/segmentio/kafka-go"
)

type Event struct {
	UserID    string    `json:"user_id"`
	Action    string    `json:"action"`
	Timestamp time.Time `json:"timestamp"`
}

func main() {
	writer := &kafka.Writer{
		Addr:     kafka.TCP("localhost:9092"),
		Topic:    "user-events",
		Balancer: &kafka.LeastBytes{},
	}
	defer writer.Close()

	users := []string{"user-1", "user-2", "user-3", "user-4", "user-5"}
	actions := []string{"login", "click", "purchase", "logout"}

	log.Println("producer: sending events every 500ms... (Ctrl+C to stop)")

	for i := 0; ; i++ {
		event := Event{
			UserID:    users[rand.Intn(len(users))],
			Action:    actions[rand.Intn(len(actions))],
			Timestamp: time.Now(),
		}

		data, _ := json.Marshal(event)

		err := writer.WriteMessages(context.Background(), kafka.Message{
			Key:   []byte(event.UserID),
			Value: data,
		})
		if err != nil {
			log.Printf("producer: write error: %v", err)
		} else {
			fmt.Printf("sent #%d: %s\n", i+1, string(data))
		}

		time.Sleep(500 * time.Millisecond)
	}
}
