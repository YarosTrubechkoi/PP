package aggregator

import (
	"context"
	"log"
	"sync"
	"time"

	"live-coding/model"
)

// Storage - интерфейс для сохранения агрегированных данных.
type Storage interface {
	UpsertCounts(ctx context.Context, counts map[string]int64) error
}

// Aggregator накапливает события в памяти и периодически сбрасывает агрегат в БД.
//
// Почему буферизация, а не запись на каждое событие:
// - При высоком потоке (тысячи событий/сек) запись каждого события в БД создаст огромную нагрузку.
// - Буферизация позволяет делать один batch upsert раз в N секунд.
//
// Влияние типа агрегации:
// - COUNT: храним map[key]int64, инкрементируем на +1. Просто и дёшево по памяти.
// - SUM:   храним map[key]float64, прибавляем event.Amount. Та же структура, другой тип.
// - AVG:   нужно хранить и сумму, и количество: map[key]{sum, count}, чтобы при flush
//          посчитать среднее = sum/count.
// - MIN/MAX: храним map[key]float64, сравниваем с текущим значением.
// - COUNT DISTINCT: храним map[key]map[distinctField]struct{}, что дороже по памяти.
type Aggregator struct {
	mu       sync.Mutex
	counts   map[string]int64 // key: userID -> value: количество событий
	storage  Storage
	interval time.Duration
}

func New(storage Storage, flushInterval time.Duration) *Aggregator {
	return &Aggregator{
		counts:   make(map[string]int64),
		storage:  storage,
		interval: flushInterval,
	}
}

// Handle вызывается на каждое событие из Kafka.
// Потокобезопасен благодаря mutex.
func (a *Aggregator) Handle(event model.Event) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Агрегация по UserID — это наш ключ группировки.
	// Если бы мы агрегировали по Action, ключом стал бы event.Action.
	// Можно делать составной ключ: userID + ":" + action — тогда агрегация по паре полей.
	a.counts[event.UserID]++
}

// Run запускает периодический flush агрегированных данных в БД.
func (a *Aggregator) Run(ctx context.Context) error {
	ticker := time.NewTicker(a.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			// Финальный flush при завершении
			a.flush(context.Background())
			return nil
		case <-ticker.C:
			a.flush(ctx)
		}
	}
}

// flush забирает накопленные данные и записывает в БД.
func (a *Aggregator) flush(ctx context.Context) {
	a.mu.Lock()
	if len(a.counts) == 0 {
		a.mu.Unlock()
		return
	}
	// Забираем текущий буфер и создаём новый — минимизируем время под локом.
	snapshot := a.counts
	a.counts = make(map[string]int64)
	a.mu.Unlock()

	log.Printf("aggregator: flushing %d user aggregates to DB", len(snapshot))

	if err := a.storage.UpsertCounts(ctx, snapshot); err != nil {
		log.Printf("aggregator: flush error: %v", err)
		// Возвращаем данные обратно, чтобы не потерять
		a.mu.Lock()
		for k, v := range snapshot {
			a.counts[k] += v
		}
		a.mu.Unlock()
	}
}
