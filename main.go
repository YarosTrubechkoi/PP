package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"live-coding/aggregator"
	"live-coding/consumer"
	"live-coding/storage"
)

func main() {
	// Конфигурация (в реальном проекте — из env/config файла)
	kafkaBrokers := []string{"localhost:9092"}
	kafkaTopic := "user-events"
	kafkaGroup := "event-aggregator"
	postgresDSN := "postgres://postgres:postgres@localhost:5432/userservice?sslmode=disable"
	flushInterval := 5 * time.Second

	// 1. Подключаемся к БД
	store, err := storage.NewPostgres(postgresDSN)
	if err != nil {
		log.Fatalf("failed to init storage: %v", err)
	}
	defer store.Close()

	// 2. Создаём агрегатор
	agg := aggregator.New(store, flushInterval)

	// 3. Создаём Kafka-консьюмер, который передаёт события в агрегатор
	cons := consumer.New(kafkaBrokers, kafkaTopic, kafkaGroup, agg.Handle)

	// 4. Graceful shutdown по SIGINT/SIGTERM
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// 5. Запускаем агрегатор (flush loop) в отдельной горутине
	go func() {
		if err := agg.Run(ctx); err != nil {
			log.Printf("aggregator stopped: %v", err)
		}
	}()

	// 6. Запускаем консьюмер (блокирующий вызов)
	log.Println("starting event aggregation service...")
	if err := cons.Run(ctx); err != nil {
		log.Fatalf("consumer error: %v", err)
	}

	log.Println("service stopped gracefully")
}
