# """
# milvus_inspect.py
# ==================
# View everything inside Milvus:
# - List collections
# - Schema (fields, dimensions)
# - Row count
# - Index info
# - Sample data preview
# """

# from pymilvus import connections, utility, Collection


# def connect():
#     connections.connect(
#         alias="default",
#         host="localhost",   # change if needed
#         port="19530"
#     )
#     print("✅ Connected to Milvus")


# def inspect_all_collections():
#     collections = utility.list_collections()

#     if not collections:
#         print("⚠️ No collections found")
#         return

#     print(f"\n📦 Total collections: {len(collections)}\n")

#     for name in collections:
#         print("=" * 60)
#         print(f"📁 Collection: {name}")
#         print("=" * 60)

#         try:
#             col = Collection(name)

#             # Load collection (needed for query)
#             col.load()

#             # 🔹 Basic info
#             print(f"🔹 Description: {col.description}")
#             print(f"🔹 Is empty: {col.is_empty}")
#             print(f"🔹 Num entities: {col.num_entities}")

#             # 🔹 Schema
#             print("\n📐 Schema:")
#             for field in col.schema.fields:
#                 print(f"   - {field.name} ({field.dtype})", end="")
#                 if hasattr(field, "params") and field.params:
#                     print(f" | params={field.params}")
#                 else:
#                     print()

#             # 🔹 Index info
#             print("\n⚙️ Indexes:")
#             if col.indexes:
#                 for idx in col.indexes:
#                     print(f"   - Field: {idx.field_name}")
#                     print(f"     Params: {idx.params}")
#             else:
#                 print("   (No index found)")

#             # 🔹 Stats (row count)
#             print("\n📊 Stats:")
#             print(f"   - Row count: {col.num_entities}")
#             print(f"   - Loaded: True")
#             print(f"   - Primary field: {col.primary_field.name}")

#             # 🔹 Sample data (first 5 rows)
#             print("\n🔍 Sample Data:")
#             try:
#                 res = col.query(expr="", limit=5)
#                 for r in res:
#                     print(f"   {r}")
#             except Exception as e:
#                 print(f"   ⚠️ Could not fetch sample data: {e}")

#         except Exception as e:
#             print(f"❌ Error inspecting {name}: {e}")

#         print("\n")


# if __name__ == "__main__":
#     connect()
#     inspect_all_collections()

# from pymilvus import connections, Collection

# connections.connect(host="localhost", port="19530")

# col = Collection("agentvigil_captions")
# col.load()

# rows = col.query(
#     expr="id >= 0",
#     output_fields=[
#         "id",
#         "pg_id",
#         "camera_id",
#         "video_name",
#         "chunk_index",
#         "event_type"
#     ],
#     limit=100
# )

# print(f"\n📦 Total rows fetched: {len(rows)}\n")

# for i, row in enumerate(rows, 1):
#     print(f"Row {i}")
#     print("-" * 50)
#     for k, v in row.items():
#         print(f"{k:<15}: {v}")
#     print()

from pymilvus import utility, connections

connections.connect("default", host="localhost", port="19530")

if utility.has_collection("agentvigil_captions"):
    utility.drop_collection("agentvigil_captions")
    print("✅ Old collection dropped")