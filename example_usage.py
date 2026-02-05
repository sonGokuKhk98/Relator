"""
Example: Using Schema Knowledge Graph for Dynamic SQL Generation

PropTech Property Search/Discovery Schema Example
- Finding properties of interest
- Listing search and filtering
- Agent/broker queries
"""

from schema_graph import SchemaGraph, Table, Column, JoinEdge


def build_proptech_schema() -> SchemaGraph:
    """Build a PropTech property search schema with semantic metadata."""

    graph = SchemaGraph()

    # ============================================================
    # Core Property Search Tables
    # ============================================================

    # Listings - properties available for sale/rent
    listings = Table(
        name="listings",
        description="Active property listings for sale or rent",
        aliases=["properties", "homes", "property", "listing", "real_estate"],
        business_terms=["property", "home", "house", "listing", "for sale", "for rent"],
        columns={
            "listing_id": Column(
                name="listing_id",
                data_type="INTEGER",
                is_primary_key=True
            ),
            "property_type_id": Column(
                name="property_type_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="property_types",
                references_column="type_id"
            ),
            "neighborhood_id": Column(
                name="neighborhood_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="neighborhoods",
                references_column="neighborhood_id"
            ),
            "agent_id": Column(
                name="agent_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="agents",
                references_column="agent_id"
            ),
            "address": Column(name="address", data_type="VARCHAR(500)"),
            "city": Column(name="city", data_type="VARCHAR(100)"),
            "state": Column(name="state", data_type="VARCHAR(50)"),
            "zip_code": Column(name="zip_code", data_type="VARCHAR(20)"),
            "price": Column(
                name="price",
                data_type="DECIMAL(12,2)",
                description="Listing price in USD"
            ),
            "bedrooms": Column(name="bedrooms", data_type="INTEGER"),
            "bathrooms": Column(name="bathrooms", data_type="DECIMAL(3,1)"),
            "square_feet": Column(name="square_feet", data_type="INTEGER"),
            "lot_size": Column(name="lot_size", data_type="DECIMAL(10,2)", description="Lot size in acres"),
            "year_built": Column(name="year_built", data_type="INTEGER"),
            "listing_type": Column(
                name="listing_type",
                data_type="VARCHAR(20)",
                description="sale, rent, lease"
            ),
            "status": Column(
                name="status",
                data_type="VARCHAR(50)",
                description="active, pending, sold, withdrawn"
            ),
            "days_on_market": Column(name="days_on_market", data_type="INTEGER"),
            "created_at": Column(name="created_at", data_type="TIMESTAMP"),
            "lat": Column(name="lat", data_type="DECIMAL(10,8)"),
            "lng": Column(name="lng", data_type="DECIMAL(11,8)")
        }
    )

    # Property Types
    property_types = Table(
        name="property_types",
        description="Types of properties (house, condo, townhouse, etc.)",
        aliases=["types", "home_types", "categories"],
        business_terms=["type", "house type", "property type"],
        columns={
            "type_id": Column(name="type_id", data_type="INTEGER", is_primary_key=True),
            "name": Column(
                name="name",
                data_type="VARCHAR(100)",
                description="single_family, condo, townhouse, multi_family, land, commercial"
            ),
            "description": Column(name="description", data_type="TEXT")
        }
    )

    # Neighborhoods
    neighborhoods = Table(
        name="neighborhoods",
        description="Neighborhood/area information with demographics",
        aliases=["areas", "locations", "districts", "zones"],
        business_terms=["neighborhood", "area", "location", "district"],
        columns={
            "neighborhood_id": Column(name="neighborhood_id", data_type="INTEGER", is_primary_key=True),
            "name": Column(name="name", data_type="VARCHAR(200)"),
            "city": Column(name="city", data_type="VARCHAR(100)"),
            "state": Column(name="state", data_type="VARCHAR(50)"),
            "median_income": Column(name="median_income", data_type="INTEGER"),
            "population": Column(name="population", data_type="INTEGER"),
            "walk_score": Column(name="walk_score", data_type="INTEGER", description="0-100 walkability score"),
            "transit_score": Column(name="transit_score", data_type="INTEGER"),
            "school_rating": Column(name="school_rating", data_type="DECIMAL(3,1)", description="1-10 school rating"),
            "crime_index": Column(name="crime_index", data_type="DECIMAL(5,2)", description="Lower is safer")
        }
    )

    # Agents/Brokers
    agents = Table(
        name="agents",
        description="Real estate agents and brokers",
        aliases=["realtors", "brokers", "agent"],
        business_terms=["agent", "realtor", "broker"],
        columns={
            "agent_id": Column(name="agent_id", data_type="INTEGER", is_primary_key=True),
            "brokerage_id": Column(
                name="brokerage_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="brokerages",
                references_column="brokerage_id"
            ),
            "first_name": Column(name="first_name", data_type="VARCHAR(100)"),
            "last_name": Column(name="last_name", data_type="VARCHAR(100)"),
            "email": Column(name="email", data_type="VARCHAR(255)"),
            "phone": Column(name="phone", data_type="VARCHAR(50)"),
            "license_number": Column(name="license_number", data_type="VARCHAR(50)"),
            "years_experience": Column(name="years_experience", data_type="INTEGER"),
            "avg_rating": Column(name="avg_rating", data_type="DECIMAL(3,2)"),
            "total_sales": Column(name="total_sales", data_type="INTEGER"),
            "specialization": Column(
                name="specialization",
                data_type="VARCHAR(100)",
                description="luxury, first_time_buyer, commercial, investment"
            )
        }
    )

    # Brokerages
    brokerages = Table(
        name="brokerages",
        description="Real estate brokerages/companies",
        aliases=["agencies", "companies", "firms"],
        business_terms=["brokerage", "agency", "company"],
        columns={
            "brokerage_id": Column(name="brokerage_id", data_type="INTEGER", is_primary_key=True),
            "name": Column(name="name", data_type="VARCHAR(255)"),
            "website": Column(name="website", data_type="VARCHAR(255)"),
            "phone": Column(name="phone", data_type="VARCHAR(50)"),
            "city": Column(name="city", data_type="VARCHAR(100)"),
            "state": Column(name="state", data_type="VARCHAR(50)")
        }
    )

    # Amenities (property features)
    amenities = Table(
        name="amenities",
        description="Property amenities and features",
        aliases=["features", "amenity"],
        business_terms=["amenity", "feature", "has"],
        columns={
            "amenity_id": Column(name="amenity_id", data_type="INTEGER", is_primary_key=True),
            "name": Column(
                name="name",
                data_type="VARCHAR(100)",
                description="pool, garage, fireplace, central_ac, hardwood_floors, etc."
            ),
            "category": Column(
                name="category",
                data_type="VARCHAR(50)",
                description="interior, exterior, community, security"
            )
        }
    )

    # Listing Amenities (junction table)
    listing_amenities = Table(
        name="listing_amenities",
        description="Links listings to their amenities",
        aliases=["property_features"],
        business_terms=[],
        columns={
            "listing_id": Column(
                name="listing_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="listings",
                references_column="listing_id"
            ),
            "amenity_id": Column(
                name="amenity_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="amenities",
                references_column="amenity_id"
            )
        }
    )

    # Saved Searches (user preferences)
    saved_searches = Table(
        name="saved_searches",
        description="User saved search criteria",
        aliases=["search_criteria", "alerts"],
        business_terms=["search", "criteria", "alert"],
        columns={
            "search_id": Column(name="search_id", data_type="INTEGER", is_primary_key=True),
            "user_id": Column(
                name="user_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="users",
                references_column="user_id"
            ),
            "min_price": Column(name="min_price", data_type="DECIMAL(12,2)"),
            "max_price": Column(name="max_price", data_type="DECIMAL(12,2)"),
            "min_bedrooms": Column(name="min_bedrooms", data_type="INTEGER"),
            "min_bathrooms": Column(name="min_bathrooms", data_type="DECIMAL(3,1)"),
            "min_sqft": Column(name="min_sqft", data_type="INTEGER"),
            "property_type_id": Column(name="property_type_id", data_type="INTEGER"),
            "city": Column(name="city", data_type="VARCHAR(100)"),
            "created_at": Column(name="created_at", data_type="TIMESTAMP")
        }
    )

    # Users (property seekers)
    users = Table(
        name="users",
        description="Users searching for properties",
        aliases=["buyers", "seekers", "customers"],
        business_terms=["user", "buyer", "customer"],
        columns={
            "user_id": Column(name="user_id", data_type="INTEGER", is_primary_key=True),
            "email": Column(name="email", data_type="VARCHAR(255)"),
            "first_name": Column(name="first_name", data_type="VARCHAR(100)"),
            "last_name": Column(name="last_name", data_type="VARCHAR(100)"),
            "phone": Column(name="phone", data_type="VARCHAR(50)"),
            "pre_approved": Column(name="pre_approved", data_type="BOOLEAN"),
            "max_budget": Column(name="max_budget", data_type="DECIMAL(12,2)"),
            "created_at": Column(name="created_at", data_type="TIMESTAMP")
        }
    )

    # Favorites (saved listings)
    favorites = Table(
        name="favorites",
        description="User favorited/saved listings",
        aliases=["saved_listings", "wishlist", "saved"],
        business_terms=["favorite", "saved", "liked"],
        columns={
            "favorite_id": Column(name="favorite_id", data_type="INTEGER", is_primary_key=True),
            "user_id": Column(
                name="user_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="users",
                references_column="user_id"
            ),
            "listing_id": Column(
                name="listing_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="listings",
                references_column="listing_id"
            ),
            "created_at": Column(name="created_at", data_type="TIMESTAMP"),
            "notes": Column(name="notes", data_type="TEXT")
        }
    )

    # Viewing Appointments
    viewings = Table(
        name="viewings",
        description="Property viewing/showing appointments",
        aliases=["showings", "appointments", "tours"],
        business_terms=["viewing", "showing", "tour", "appointment"],
        columns={
            "viewing_id": Column(name="viewing_id", data_type="INTEGER", is_primary_key=True),
            "listing_id": Column(
                name="listing_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="listings",
                references_column="listing_id"
            ),
            "user_id": Column(
                name="user_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="users",
                references_column="user_id"
            ),
            "agent_id": Column(
                name="agent_id",
                data_type="INTEGER",
                is_foreign_key=True,
                references_table="agents",
                references_column="agent_id"
            ),
            "scheduled_at": Column(name="scheduled_at", data_type="TIMESTAMP"),
            "status": Column(
                name="status",
                data_type="VARCHAR(50)",
                description="scheduled, completed, cancelled, no_show"
            ),
            "feedback": Column(name="feedback", data_type="TEXT")
        }
    )

    # Add all tables
    tables = [
        listings, property_types, neighborhoods, agents, brokerages,
        amenities, listing_amenities, saved_searches, users, favorites, viewings
    ]
    for table in tables:
        graph.add_table(table)

    # ============================================================
    # Define Relationships
    # ============================================================

    relationships = [
        JoinEdge("listings", "property_type_id", "property_types", "type_id", "many-to-one"),
        JoinEdge("listings", "neighborhood_id", "neighborhoods", "neighborhood_id", "many-to-one"),
        JoinEdge("listings", "agent_id", "agents", "agent_id", "many-to-one"),
        JoinEdge("agents", "brokerage_id", "brokerages", "brokerage_id", "many-to-one"),
        JoinEdge("listing_amenities", "listing_id", "listings", "listing_id", "many-to-one"),
        JoinEdge("listing_amenities", "amenity_id", "amenities", "amenity_id", "many-to-one"),
        JoinEdge("saved_searches", "user_id", "users", "user_id", "many-to-one"),
        JoinEdge("favorites", "user_id", "users", "user_id", "many-to-one"),
        JoinEdge("favorites", "listing_id", "listings", "listing_id", "many-to-one"),
        JoinEdge("viewings", "listing_id", "listings", "listing_id", "many-to-one"),
        JoinEdge("viewings", "user_id", "users", "user_id", "many-to-one"),
        JoinEdge("viewings", "agent_id", "agents", "agent_id", "many-to-one"),
    ]

    for rel in relationships:
        graph.add_relationship(rel)

    return graph


def demo_join_path_finding():
    """Demonstrate finding join paths between tables."""

    print("=" * 60)
    print("PropTech Property Search - Knowledge Graph Demo")
    print("=" * 60)

    graph = build_proptech_schema()

    # ============================================================
    # Example 1: Simple join - listings to neighborhoods
    # ============================================================
    print("\n1. Find path: listings -> neighborhoods")
    print("-" * 40)

    path = graph.find_join_path("listings", "neighborhoods")
    if path:
        for from_t, to_t, edge in path:
            print(f"   {from_t} --> {to_t}")
            print(f"   JOIN ON: {edge['join_condition']}")

    # ============================================================
    # Example 2: Multi-hop - users to brokerages (via favorites, listings, agents)
    # ============================================================
    print("\n2. Find path: users -> brokerages")
    print("-" * 40)

    path = graph.find_join_path("users", "brokerages")
    if path:
        print(f"   Path found with {len(path)} joins:")
        for from_t, to_t, edge in path:
            print(f"   {from_t} --> {to_t}")

    # ============================================================
    # Example 3: Using aliases
    # ============================================================
    print("\n3. Using aliases: 'homes' -> 'areas'")
    print("-" * 40)

    path = graph.find_join_path("homes", "areas")
    if path:
        print("   Resolved: 'homes' -> listings, 'areas' -> neighborhoods")
        for from_t, to_t, edge in path:
            print(f"   {from_t} --> {to_t}")

    # ============================================================
    # Example 4: Generate JOIN SQL for property search with amenities
    # ============================================================
    print("\n4. Generate JOIN SQL: listings + amenities + neighborhoods")
    print("-" * 40)

    sql = graph.generate_join_sql(["listings", "amenities", "neighborhoods"])
    print(sql)

    # ============================================================
    # Example 5: Related tables discovery
    # ============================================================
    print("\n5. Tables related to 'listings' (within 2 hops)")
    print("-" * 40)

    related = graph.get_related_tables("listings", max_distance=2)
    for table, distance in sorted(related.items(), key=lambda x: x[1]):
        print(f"   {table}: {distance} hop(s) away")


def demo_property_search_queries():
    """
    Real-world property search query examples.
    """

    print("\n" + "=" * 60)
    print("Property Search Query Examples")
    print("=" * 60)

    graph = build_proptech_schema()

    # ============================================================
    # Query 1: Find properties with specific amenities
    # ============================================================
    print("\nQuery 1: 'Find 3+ bedroom homes with pool in safe neighborhoods'")
    print("-" * 55)

    tables = ["listings", "listing_amenities", "amenities", "neighborhoods"]
    join_sql = graph.generate_join_sql(tables)

    query1 = f"""
SELECT
    listings.address,
    listings.city,
    listings.price,
    listings.bedrooms,
    listings.bathrooms,
    listings.square_feet,
    neighborhoods.name AS neighborhood,
    neighborhoods.school_rating,
    neighborhoods.crime_index
FROM
{join_sql}
WHERE
    listings.bedrooms >= 3
    AND listings.status = 'active'
    AND amenities.name = 'pool'
    AND neighborhoods.crime_index < 50
ORDER BY
    listings.price ASC
"""
    print(query1)

    # ============================================================
    # Query 2: Top agents by neighborhood
    # ============================================================
    print("\nQuery 2: 'Top rated agents in Downtown with most listings'")
    print("-" * 55)

    tables = ["agents", "listings", "neighborhoods", "brokerages"]
    join_sql = graph.generate_join_sql(tables)

    query2 = f"""
SELECT
    agents.first_name,
    agents.last_name,
    brokerages.name AS brokerage,
    agents.avg_rating,
    agents.years_experience,
    COUNT(listings.listing_id) AS active_listings
FROM
{join_sql}
WHERE
    neighborhoods.name LIKE '%Downtown%'
    AND listings.status = 'active'
GROUP BY
    agents.agent_id, agents.first_name, agents.last_name,
    brokerages.name, agents.avg_rating, agents.years_experience
ORDER BY
    agents.avg_rating DESC, active_listings DESC
LIMIT 10
"""
    print(query2)

    # ============================================================
    # Query 3: User's favorited properties with details
    # ============================================================
    print("\nQuery 3: 'Show user's saved properties with neighborhood info'")
    print("-" * 55)

    tables = ["users", "favorites", "listings", "neighborhoods", "property_types"]
    join_sql = graph.generate_join_sql(tables)

    query3 = f"""
SELECT
    users.email,
    listings.address,
    listings.price,
    property_types.name AS property_type,
    listings.bedrooms,
    neighborhoods.name AS neighborhood,
    neighborhoods.walk_score,
    favorites.notes,
    favorites.created_at AS saved_on
FROM
{join_sql}
WHERE
    users.user_id = :user_id
ORDER BY
    favorites.created_at DESC
"""
    print(query3)

    # ============================================================
    # Query 4: Properties matching saved search criteria
    # ============================================================
    print("\nQuery 4: 'Find properties matching user's saved search'")
    print("-" * 55)

    tables = ["saved_searches", "users", "listings", "property_types"]
    join_sql = graph.generate_join_sql(tables)

    query4 = f"""
SELECT
    listings.listing_id,
    listings.address,
    listings.price,
    listings.bedrooms,
    listings.bathrooms,
    listings.square_feet,
    property_types.name AS type
FROM
{join_sql}
WHERE
    saved_searches.user_id = :user_id
    AND listings.price BETWEEN saved_searches.min_price AND saved_searches.max_price
    AND listings.bedrooms >= saved_searches.min_bedrooms
    AND listings.square_feet >= saved_searches.min_sqft
    AND listings.status = 'active'
ORDER BY
    listings.created_at DESC
"""
    print(query4)

    # ============================================================
    # Query 5: Viewing schedule with property and agent details
    # ============================================================
    print("\nQuery 5: 'Show upcoming viewings with property and agent info'")
    print("-" * 55)

    tables = ["viewings", "listings", "agents", "users", "neighborhoods"]
    join_sql = graph.generate_join_sql(tables)

    query5 = f"""
SELECT
    viewings.scheduled_at,
    listings.address,
    listings.price,
    neighborhoods.name AS neighborhood,
    agents.first_name || ' ' || agents.last_name AS agent_name,
    agents.phone AS agent_phone,
    users.first_name || ' ' || users.last_name AS buyer_name
FROM
{join_sql}
WHERE
    viewings.status = 'scheduled'
    AND viewings.scheduled_at > NOW()
ORDER BY
    viewings.scheduled_at ASC
"""
    print(query5)


def demo_schema_visualization():
    """Show the schema graph."""
    print("\n" + "=" * 60)
    print("Schema Graph Visualization")
    print("=" * 60)

    graph = build_proptech_schema()
    print(graph.visualize())


if __name__ == "__main__":
    demo_join_path_finding()
    demo_property_search_queries()
    demo_schema_visualization()
