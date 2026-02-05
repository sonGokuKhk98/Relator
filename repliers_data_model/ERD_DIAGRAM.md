# Repliers Real Estate Data Model - ERD

## Entity Relationship Diagram

```mermaid
erDiagram
    DIM_STATES {
        int state_id PK
        string state_code UK
        string state_name
        string region
    }
    
    DIM_CITIES {
        int city_id PK
        string city_name
        int state_id FK
        string state_code
        string county_area
    }
    
    DIM_NEIGHBORHOODS {
        int neighborhood_id PK
        string neighborhood_name
        int city_id FK
        string city_name
        string state_code
    }
    
    DIM_LOCATIONS {
        int location_id PK
        string location_code UK
        string location_name
        string location_type
        string area
        string city
        string state_code FK
        string country
        float latitude
        float longitude
    }
    
    DIM_PROPERTY_TYPES {
        int property_type_id PK
        string property_type_name UK
        string category
    }
    
    DIM_LISTING_STATUS {
        int status_id PK
        string status_code UK
        string status_name
        boolean is_active
        boolean is_sold
    }
    
    FACT_LISTINGS {
        int listing_id PK
        string mls_number UK
        int board_id
        string property_class
        string address_key
        string street_number
        string street_name
        string street_suffix
        string unit_number
        string city
        string state_code FK
        string zip_code
        string neighborhood
        string county_area
        int city_id FK
        int property_type_id FK
        int status_id FK
        int bedrooms
        float bathrooms
        int sqft
        string property_type
        float lot_size
        string lot_measurement
        float lot_acres
        float lot_sqft
        string lot_features
        decimal list_price
        decimal sold_price
        datetime list_date
        datetime sold_date
        datetime updated_on
        string status
        string last_status
        float latitude
        float longitude
        int image_count
    }
    
    FACT_LISTING_IMAGES {
        int image_id PK
        int listing_id FK
        string mls_number
        string image_url
        int image_order
    }
    
    AGG_STATE_STATS {
        string state_code PK
        int listing_count
    }
    
    AGG_CITY_STATS {
        string city_name PK
        int listing_count
    }
    
    AGG_PROPERTY_TYPE_STATS {
        string property_type PK
        int listing_count
    }
    
    AGG_STATUS_STATS {
        string status_code PK
        string status_name
        int listing_count
    }
    
    DIM_STATES ||--o{ DIM_CITIES : "has"
    DIM_CITIES ||--o{ DIM_NEIGHBORHOODS : "contains"
    DIM_STATES ||--o{ DIM_LOCATIONS : "contains"
    DIM_STATES ||--o{ FACT_LISTINGS : "has listings in"
    DIM_CITIES ||--o{ FACT_LISTINGS : "has listings"
    DIM_PROPERTY_TYPES ||--o{ FACT_LISTINGS : "categorizes"
    DIM_LISTING_STATUS ||--o{ FACT_LISTINGS : "describes"
    FACT_LISTINGS ||--o{ FACT_LISTING_IMAGES : "has"
```

## Table Descriptions

### Dimension Tables

| Table | Description | Row Count |
|-------|-------------|-----------|
| `dim_states` | US States reference with regions | ~17 |
| `dim_cities` | Cities with state relationships | ~800+ |
| `dim_neighborhoods` | Neighborhoods within cities | ~2000+ |
| `dim_locations` | Geographic areas (counties, areas) | ~300 |
| `dim_property_types` | Property type classifications | ~10 |
| `dim_listing_status` | Listing status codes | ~12 |

### Fact Tables

| Table | Description | Row Count |
|-------|-------------|-----------|
| `fact_listings` | Main property listings (denormalized) | ~19,000+ |
| `fact_listing_images` | Property images (one row per image) | ~400,000+ |

### Aggregate Tables

| Table | Description |
|-------|-------------|
| `agg_state_stats` | Listings count by state |
| `agg_city_stats` | Listings count by city |
| `agg_property_type_stats` | Listings count by property type |
| `agg_status_stats` | Listings count by status |

## Key Relationships

1. **State → City → Neighborhood**: Geographic hierarchy
2. **Listing → Property Type**: What type of property
3. **Listing → Status**: Current listing status
4. **Listing → Images**: One-to-many relationship (avg 20 images per listing)

## Sample Queries

### Top 10 Cities by Listing Count
```sql
SELECT c.city_name, c.state_code, COUNT(*) as listings
FROM fact_listings l
JOIN dim_cities c ON l.city_id = c.city_id
GROUP BY c.city_name, c.state_code
ORDER BY listings DESC
LIMIT 10;
```

### Average Price by State
```sql
SELECT s.state_name, s.region,
       AVG(l.list_price) as avg_price,
       COUNT(*) as listings
FROM fact_listings l
JOIN dim_states s ON l.state_code = s.state_code
WHERE l.list_price > 0
GROUP BY s.state_name, s.region
ORDER BY avg_price DESC;
```

### Property Distribution by Type
```sql
SELECT pt.property_type_name, pt.category,
       COUNT(*) as listings,
       AVG(l.list_price) as avg_price
FROM fact_listings l
JOIN dim_property_types pt ON l.property_type_id = pt.property_type_id
GROUP BY pt.property_type_name, pt.category
ORDER BY listings DESC;
```
