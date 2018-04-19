package edu.usc.thevillagers.serversideagent.recording;

import java.io.File;
import java.io.IOException;
import java.util.Map.Entry;

import edu.usc.thevillagers.serversideagent.recording.event.RecordEvent;
import edu.usc.thevillagers.serversideagent.recording.event.RecordEventEntitySpawn;
import net.minecraft.block.Block;
import net.minecraft.block.state.IBlockState;
import net.minecraft.entity.EntityList;
import net.minecraft.nbt.NBTBase;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.nbt.NBTTagList;
import net.minecraft.util.math.BlockPos;
import net.minecraftforge.common.util.Constants;

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
	
	public void setDataFromWorld(WorldRecord wr) {
		BlockPos diff = wr.to.subtract(wr.from).add(1, 1, 1);
		data = new SnapshotData(diff.getX() * diff.getY() * diff.getZ());
		int index = 0;
		for(BlockPos p : BlockPos.getAllInBoxMutable(wr.from, wr.to)) {
			data.blockStates[index++] = wr.world.getBlockState(p);
		}
		
		for(Entry<Integer, NBTTagCompound> entry : wr.computeEntitiesData(wr.getRecordWorld()).entrySet()) {
			int type = EntityList.getID(wr.getRecordWorld().getEntityByID(entry.getKey()).getClass());
			data.spawnEvents.add(new RecordEventEntitySpawn(entry.getKey(), type, entry.getValue()));
		}
	}
	
	public void applyDataToWorld(WorldRecord wr) {
		int index = 0;
		ReplayWorldAccess world = wr.getReplayWorld();
		world.reset();
		for(BlockPos p : BlockPos.getAllInBoxMutable(wr.from, wr.to)) {
			world.setBlockState(p, data.blockStates[index++]);
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
	}
}
