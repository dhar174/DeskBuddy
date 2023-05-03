import asyncio
from picoh import picoh


async def test_lip_movement():
    await picoh.reset()
    await asyncio.sleep(1)

    for i in range(2, 8):
        await picoh.move(picoh.TOPLIP, i)
        await picoh.move(picoh.BOTTOMLIP, i)
        await asyncio.sleep(0.5)

    await picoh.move(picoh.TOPLIP, 5)
    await picoh.move(picoh.BOTTOMLIP, 5)
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_lip_movement())
