package consumer

import (
	"context"
	"encoding/json"
	"log"

	"github.com/segmentio/kafka-go"

	"live-coding/model"
)

type EventHandler func(event model.Event)

type Consumer struct {
	reader  *kafka.Reader
	handler EventHandler
}

func New(brokers []string, topic, groupID string, handler EventHandler) *Consumer {
	reader := kafka.NewReader(kafka.ReaderConfig{
		Brokers: brokers,
		Topic:   topic,
		GroupID: groupID,
	})

	return &Consumer{
		reader:  reader,
		handler: handler,
	}
}

// Run читает сообщения из Kafka и передаёт их в handler.
// Блокирующий вызов — работает пока не отменён ctx.
func (c *Consumer) Run(ctx context.Context) error {
	defer c.reader.Close()

	log.Println("consumer: started, waiting for messages...")

	for {
		msg, err := c.reader.ReadMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				log.Println("consumer: context cancelled, shutting down")
				return nil
			}
			log.Printf("consumer: read error: %v", err)
			continue
		}

		var event model.Event
		if err := json.Unmarshal(msg.Value, &event); err != nil {
			log.Printf("consumer: unmarshal error: %v, raw: %s", err, string(msg.Value))
			continue
		}

		c.handler(event)
	}
}
