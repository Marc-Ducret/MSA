package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Map.Entry;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntitySpawn;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventTileEntitySpawn;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import net.minecraft.block.Block;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.multiplayer.ChunkProviderClient;
import net.minecraft.entity.EntityList;
import net.minecraft.init.Blocks;
import net.minecraft.nbt.NBTBase;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.nbt.NBTTagList;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.storage.ExtendedBlockStorage;
import net.minecraftforge.common.util.Constants;

/**
 * File interface for a complete state of the world at some point in time.
 */
public class Snapshot extends NBTFileInterface<SnapshotData> {
	
	public Snapshot(File file) {
		super(file);
	}
	
	private int encodeBlockState(IBlockState state) {
		return Block.getStateId(state);
	}
	
	private IBlockState decodeBlockState(int id) {
		return Block.getStateById(id);
	}
	
	public void setDataFromWorld(WorldRecordRecorder wr) {
		BlockPos diff = wr.to.subtract(wr.from).add(1, 1, 1);
		data = new SnapshotData(diff.getX() * diff.getY() * diff.getZ());
		int index = 0;
		for(BlockPos p : BlockPos.getAllInBoxMutable(wr.from, wr.to)) {
			data.blockStates[index++] = wr.world.getBlockState(p);
		}
		
		for(Entry<Integer, NBTTagCompound> entry : wr.computeEntitiesData(wr.world).entrySet()) {
			int type = EntityList.getID(wr.world.getEntityByID(entry.getKey()).getClass());
			data.spawnEvents.add(new RecordEventEntitySpawn(entry.getKey(), type, entry.getValue()));
		}
		for(Entry<BlockPos, NBTTagCompound> entry : wr.computeTileEntitiesData(wr.world).entrySet()) {
			data.spawnEvents.add(new RecordEventTileEntitySpawn(entry.getKey(), entry.getValue()));
		}
		data.worldTime = wr.world.getTotalWorldTime();
	}
	
	protected int index(int x, int y, int z, BlockPos from, BlockPos diff) {
		int dx = x - from.getX();
		int dy = y - from.getY();
		int dz = z - from.getZ();
		if(		dx < 0 || dx >= diff.getX() ||
				dy < 0 || dy >= diff.getY() ||
				dz < 0 || dz >= diff.getZ())
			return -1;
		return dx + diff.getX() * (dy + diff.getY() * dz);
	}
	
	public void applyDataToWorld(WorldRecordReplayer wr) {
		BlockPos diff = wr.to.subtract(wr.from).add(1, 1, 1);
		try {
			ChunkProviderClient chunkProvider = (ChunkProviderClient) wr.world.getChunkProvider();
			Field chunkMappingField = ChunkProviderClient.class.getDeclaredField("chunkMapping");
			chunkMappingField.setAccessible(true);
			Field storageField = Chunk.class.getDeclaredField("storageArrays");
			storageField.setAccessible(true);
			Long2ObjectMap<Chunk> chunkMapping = (Long2ObjectMap<Chunk>) chunkMappingField.get(chunkProvider);
			int margin = 1;
			for(int chunkZ = (wr.from.getZ() >> 4) - margin; chunkZ <= (wr.to.getZ() >> 4) + margin; chunkZ++)
				for(int chunkX = (wr.from.getX() >> 4) - margin; chunkX <= (wr.to.getX() >> 4) + margin; chunkX++) {
					long chunkPos = ChunkPos.asLong(chunkX, chunkZ);
					Chunk c;
					if(!chunkMapping.containsKey(chunkPos)) {
						c = new Chunk(wr.world, chunkX, chunkZ);
						chunkMapping.put(chunkPos, c);
					} else c = chunkMapping.get(chunkPos);
					ExtendedBlockStorage[] storage = (ExtendedBlockStorage[]) storageField.get(c);
					for(int y = 0; y < 0x100; y++) {
						for(int z = 0; z < 0x10; z++) {
							for(int x = 0; x < 0x10; x++) {
								int index = index(x + (chunkX << 4), y, z + (chunkZ << 4), wr.from, diff);
								if(index >= 0) {
									IBlockState state = data.blockStates[index];
									if(state.getBlock() != Blocks.AIR) {
										if(storage[y >> 4] == null) storage[y >> 4] = new ExtendedBlockStorage((y >> 4) << 4, true);
										storage[y >> 4].set(x, y & 0xF, z, state);
										storage[y >> 4].setBlockLight(x, y & 0xF, z, 0x8);
										storage[y >> 4].setSkyLight(x, y & 0xF, z, 0xF);
									}
								}
							}
						}
					}
					c.markLoaded(true);
					c.generateSkylightMap();
				}
			wr.world.markBlockRangeForRenderUpdate(wr.from, wr.to);
		} catch (Exception e) {
			e.printStackTrace();
		}
		for(RecordEvent event : data.spawnEvents) event.replay(wr);
	}

	@Override
	protected void writeNBT(NBTTagCompound compound) throws IOException {
		int[] encodedData = new int[data.blockStates.length];
		for(int i = 0; i < data.blockStates.length; i ++)
			encodedData[i] = encodeBlockState(data.blockStates[i]);
		compound.setIntArray("BlockStates", encodedData);
		
		NBTTagList list = new NBTTagList();
		for(RecordEvent event : data.spawnEvents) {
			list.appendTag(RecordEvent.toNBT(event));
		}
		compound.setTag("SpawnEvents", list);
		compound.setLong("WorldTime", data.worldTime);
	}

	@Override
	protected void readNBT(NBTTagCompound compound) throws IOException {
		int[] encodedData = compound.getIntArray("BlockStates");
		if(encodedData == null) throw new IOException("Wrong file format");
		data = new SnapshotData(encodedData.length);
		for(int i = 0; i < data.blockStates.length; i++)
			data.blockStates[i] = decodeBlockState(encodedData[i]);
		
		NBTTagList list = compound.getTagList("SpawnEvents", Constants.NBT.TAG_COMPOUND);
		for(NBTBase nbt : list) {
			data.spawnEvents.add(RecordEvent.fromNBT((NBTTagCompound) nbt));
		}
		data.worldTime = compound.getLong("WorldTime");
	}
}
