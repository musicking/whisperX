from httpx import AsyncClient


async def test_health_and_capabilities(client: AsyncClient) -> None:
    live = await client.get("/health/live")
    ready = await client.get("/health/ready")
    capabilities = await client.get("/v1/capabilities")

    assert live.status_code == 200
    assert ready.status_code == 200
    assert capabilities.status_code == 200
    assert capabilities.json()["realtime_streaming"] is False
    assert "transcribe" in capabilities.json()["tasks"]


async def test_models_are_allowlisted(client: AsyncClient) -> None:
    response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert any(model["default"] for model in response.json()["data"])
