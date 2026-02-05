# 🧪 Dynamic Schema Graph - Test Prompts

Test your PropTech SQL generation system with these creative prompts. 
Each section targets a specific feature.

---

## 📊 Basic Filters

### Price Filters
```
homes under 500k
properties less than 1 million dollars
listings above $2m
houses between 300k and 800k
cheap homes under 200000
luxury properties over 5 million
budget friendly options max 350k
```

### Bedroom/Bathroom Filters
```
3 bedroom houses
4+ bedroom homes
2 bed 2 bath condos
homes with at least 5 bedrooms
single bedroom apartments
5 br properties
homes with 3 bathrooms
```

### Square Footage
```
properties over 2000 sqft
homes less than 1500 square feet
large houses above 3000 sft
small apartments under 800 sq ft
spacious 2500+ sqft listings
compact units below 600 sft
```

---

## 🔄 Unit Conversion Tests

These specifically test k/m/sft parsing:

```
homes under 2m with 3000 sqft
500k budget for 1500 sft minimum
properties 750k-1.5m range
2 million dollar homes over 4000 square feet
listings under 1.2m
homes around 800k with 2000+ sft
cheap 400k houses with at least 1800 sqft
```

---

## 🏊 Amenity Searches

```
homes with a pool
properties with garage and fireplace
houses with swimming pool in downtown
luxury homes with gym and parking
listings with garden
properties with 2 car garage
homes with pool and at least 4 bedrooms
waterfront properties with dock
pet friendly homes with fenced yard
smart homes with security system
```

---

## 📍 Location & Neighborhood

```
downtown condos
properties in walkable areas
homes near good schools
suburban houses with low crime
listings in transit-friendly neighborhoods
properties in safe neighborhoods
urban apartments downtown
homes in family-friendly areas
listings with high walk score
properties in gated communities
```

---

## 🏠 Property Types

```
single family homes
condos for sale
townhouses under 600k
multi-family properties
studio apartments
loft spaces downtown
duplex investments
colonial style houses
modern condos with balcony
victorian homes
```

---

## 🔀 Complex Multi-Filter Queries

These combine multiple conditions:

```
3 bed 2 bath homes under 500k with pool in downtown
luxury condos over 2m with 2000+ sqft and gym
family homes with 4 bedrooms near good schools under 800k
modern townhouses with garage and fireplace below 650k
waterfront properties with pool over 3000 sqft
walkable downtown condos under 1.5m with parking
investment properties with 3+ bedrooms in suburban areas
pet friendly homes with yard under 400k near transit
starter homes under 300k with at least 2 bedrooms
retirement condos with gym and pool in safe neighborhoods
```

---

## 🌍 Natural Language Variations

Test how well the system handles different phrasings:

```
I'm looking for a 3 bedroom house
show me properties that cost less than 500 thousand
find me homes with a swimming pool
what's available under 1 mil with 4 beds?
need a condo downtown around 600k
looking to spend about 750k on a family home
can you show luxury listings over 2 million?
I want at least 2000 square feet for under 800k
budget is 450k, need 3 beds and pool
first time buyer - something under 350k with 2 br
```

---

## 🤔 Edge Cases & Tricky Inputs

```
homes
show me everything
cheap properties
nice houses in good areas
something modern and spacious
luxury
investment opportunity
fixer upper under 200k
new construction
recently renovated homes under 1m
```

---

## 🔗 Multi-Table Join Tests

These should trigger complex graph traversals:

```
listings by top rated agents from premier brokerages
homes viewed by more than 10 users
most favorited properties this month
properties with agent contact info and neighborhood stats
listings with all amenities and property type details
homes where scheduled viewings exist
popular agents with listings in downtown
saved searches matching current listings
user favorites with full property details
brokerages with available listings under 500k
```

---

## 📈 Sorting & Ordering Tests

```
cheapest homes first
most expensive properties
sort by price ascending
newest listings
largest homes by square feet
best neighborhoods by walk score
highest rated agents
properties ordered by bedrooms descending
recently added listings
oldest properties on market
```

---

## ❓ Ambiguous Queries (Stress Tests)

See how the system handles ambiguity:

```
big house
nice area
good price
something spacious
modern amenities
great investment
perfect family home
dream property
affordable luxury
hidden gem
```

---

## 🎯 Expected Behavior Checklist

For each prompt, verify:

- [ ] **Correct tables detected** (listings, amenities, neighborhoods, etc.)
- [ ] **Proper JOIN paths** generated
- [ ] **Unit conversions** applied (k→000, m→000000, sft→square_feet)
- [ ] **Operator inference** correct (<, >, =, LIKE)
- [ ] **Filter values** parsed accurately
- [ ] **SQL syntax** is valid and executable

---

## 🚀 Speed Run Challenge

Try these in rapid succession to test real-time updates:

1. `homes` (base query)
2. `homes downtown` (add location)
3. `homes downtown with pool` (add amenity)
4. `3 bed homes downtown with pool` (add bedrooms)
5. `3 bed homes downtown with pool under 800k` (add price)
6. `3 bed homes downtown with pool under 800k over 2000 sqft` (add size)

Watch the SQL evolve in real-time!

---

## 📝 Notes

- **If LLM is active**: Should understand natural language nuances
- **If using fallback**: Relies on keyword matching, may be less accurate
- **Check the console**: Server logs show parsing details (`🤖 Gemini parsed:`)

Happy Testing! 🎉
