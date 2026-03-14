from pymilvus import connections, utility

connections.connect("default", host="localhost", port="19530")
if utility.has_collection("agentvigil_captions"):
    utility.drop_collection("agentvigil_captions")
    print("Dropped old collection")
connections.disconnect("default")