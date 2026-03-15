package storage

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	_ "github.com/lib/pq"
)

type Postgres struct {
	db *sql.DB
}

func NewPostgres(dsn string) (*Postgres, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("storage: open db: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("storage: ping db: %w", err)
	}

	if err := migrate(db); err != nil {
		return nil, fmt.Errorf("storage: migrate: %w", err)
	}

	return &Postgres{db: db}, nil
}

func migrate(db *sql.DB) error {
	query := `
		CREATE TABLE IF NOT EXISTS user_event_counts (
			user_id    TEXT PRIMARY KEY,
			count      BIGINT NOT NULL DEFAULT 0,
			updated_at TIMESTAMP NOT NULL DEFAULT now()
		);
	`
	_, err := db.Exec(query)
	return err
}

// UpsertCounts записывает агрегированные счётчики в БД одним batch-запросом.
// Использует INSERT ... ON CONFLICT (upsert), чтобы:
// - создать запись, если пользователя ещё нет
// - прибавить к существующему count, если запись уже есть
//
// Для других типов агрегации upsert менялся бы:
// - SUM:  SET amount = user_event_counts.amount + EXCLUDED.amount
// - MAX:  SET amount = GREATEST(user_event_counts.amount, EXCLUDED.amount)
// - AVG:  нужно хранить sum и count отдельно, пересчитывать при каждом upsert
func (p *Postgres) UpsertCounts(ctx context.Context, counts map[string]int64) error {
	if len(counts) == 0 {
		return nil
	}

	// Строим batch upsert: INSERT INTO ... VALUES (...), (...) ON CONFLICT ...
	var (
		placeholders []string
		args         []interface{}
		i            = 1
	)

	for userID, count := range counts {
		placeholders = append(placeholders, fmt.Sprintf("($%d, $%d, now())", i, i+1))
		args = append(args, userID, count)
		i += 2
	}

	query := fmt.Sprintf(`
		INSERT INTO user_event_counts (user_id, count, updated_at)
		VALUES %s
		ON CONFLICT (user_id) DO UPDATE SET
			count = user_event_counts.count + EXCLUDED.count,
			updated_at = now()
	`, strings.Join(placeholders, ", "))

	_, err := p.db.ExecContext(ctx, query, args...)
	return err
}

func (p *Postgres) Close() error {
	return p.db.Close()
}
