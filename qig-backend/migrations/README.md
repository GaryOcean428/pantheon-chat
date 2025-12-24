# QIG Backend Migrations

## Populate Vocabulary

To fix the "BPE garble" issue where kernels produce unreadable output,
run the vocabulary population script:

```bash
# Set your database URL
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"

# Run the migration
psql $DATABASE_URL -f populate_bip39_vocabulary.sql
```

This will:
1. Create the `tokenizer_vocabulary` table if it doesn't exist
2. Add 200+ BIP39 words with source_type='bip39'
3. Add common QIG terms with source_type='base'
4. Generate deterministic 64D basin embeddings for each word

## Verify

Check the vocabulary was populated:

```sql
SELECT source_type, COUNT(*) FROM tokenizer_vocabulary GROUP BY source_type;
```

Expected output:
```
 source_type | count
-------------+-------
 bip39       |  200+
 base        |   15+
```

## Fallback

If the database is unavailable, the `PostgresCoordizer` will automatically
use the built-in fallback vocabulary (2048 BIP39 words) to ensure
readable output.
