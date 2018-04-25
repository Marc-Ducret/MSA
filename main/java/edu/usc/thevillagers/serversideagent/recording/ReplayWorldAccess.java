package edu.usc.thevillagers.serversideagent.recording;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import net.minecraft.block.state.IBlockState;
import net.minecraft.entity.Entity;
import net.minecraft.entity.player.EntityPlayer;
import net.minecraft.init.Blocks;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.GameType;
import net.minecraft.world.IBlockAccess;
import net.minecraft.world.World;
import net.minecraft.world.WorldServer;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.WorldType;
import net.minecraft.world.biome.Biome;
import net.minecraft.world.storage.WorldInfo;
import net.minecraftforge.fml.common.FMLCommonHandler;

public class ReplayWorldAccess implements IBlockAccess {

	public final BlockPos from, to, diff;

	protected IBlockState[] blockBuffer;
	private final Map<Integer, Entity> entities;
	private final Map<BlockPos, TileEntity> tileEntities;
	
	public World fakeWorld;

	public ReplayWorldAccess(BlockPos from, BlockPos to) {
		this.from = from.toImmutable();
		this.to = to.toImmutable();

		diff = to.subtract(from).add(1, 1, 1);
		blockBuffer = new IBlockState[diff.getX() * diff.getY() * diff.getZ()];
		entities = new HashMap<>();
		tileEntities = new HashMap<>();
		
		fakeWorld = createFakeWorld();
	}
	
	public World createFakeWorld() {
		WorldSettings settings = new WorldSettings(0, GameType.NOT_SET, false, false, WorldType.FLAT);
		WorldInfo info = new WorldInfo(settings, "dummy");
		
		return new WorldServer(FMLCommonHandler.instance().getMinecraftServerInstance(), 
				FMLCommonHandler.instance().getMinecraftServerInstance().getActiveAnvilConverter().getSaveLoader("dummy", false),
				info, 0, FMLCommonHandler.instance().getMinecraftServerInstance().profiler);
	}

	@Override
	public TileEntity getTileEntity(BlockPos pos) {
		return tileEntities.get(pos);
	}

	@Override
	public int getCombinedLight(BlockPos pos, int lightValue) {
		return 0xF << 4;
	}

	protected int index(BlockPos pos) {
		BlockPos p = pos.subtract(from);
		if(		p.getX() < 0 || p.getX() >= diff.getX() ||
				p.getY() < 0 || p.getY() >= diff.getY() ||
				p.getZ() < 0 || p.getZ() >= diff.getZ())
			return -1;
		return p.getX() + diff.getX() * (p.getY() + diff.getY() * p.getZ());
	}

	@Override
	public IBlockState getBlockState(BlockPos pos) {
		int index = index(pos);
		if(index < 0) return Blocks.AIR.getDefaultState();
		return blockBuffer[index];
	}

	public void setBlockState(BlockPos pos, IBlockState state) {
		int index = index(pos);
		if(index >= 0 && !blockBuffer[index].equals(state)) {
			blockBuffer[index] = state;
		}
	}

	public void setBlockStates(IBlockState[] buffer) {
		System.arraycopy(buffer, 0, blockBuffer, 0, blockBuffer.length);
	}

	@Override
	public boolean isAirBlock(BlockPos pos) {
		return getBlockState(pos).getBlock() == Blocks.AIR;
	}

	@Override
	public Biome getBiome(BlockPos pos) {
		return Biome.getBiome(1);
	}

	@Override
	public int getStrongPower(BlockPos pos, EnumFacing direction) {
		return 0;
	}

	@Override
	public WorldType getWorldType() {
		return WorldType.CUSTOMIZED;
	}

	@Override
	public boolean isSideSolid(BlockPos pos, EnumFacing side, boolean _default) {
		int index = index(pos);
		if(index < 0) return _default;
		return blockBuffer[index].isSideSolid(this, pos, side);
	}

	public void spawnEntity(Entity e) {
		entities.put(e.getEntityId(), e);
	}

	public void killEntity(int id) {
		entities.remove(id);
	}

	public void updateEntity(int id, NBTTagCompound data) {
		if(entities.containsKey(id)) {
			Entity e = entities.get(id);
			e.readFromNBT(data);
			if(e instanceof EntityPlayer)
				e.setSneaking(data.getBoolean("Sneaking"));
		}
		else System.out.println("Missing entity with id: "+id);
	}

	public Collection<Entity> getEntities() {
		return entities.values();
	}

	public Entity getEntity(int entityId) {
		return entities.get(entityId);
	}

	public void spawnTileEntity(TileEntity tileEntity) {
		tileEntities.put(tileEntity.getPos(), tileEntity);
	}

	public void killTileEntity(BlockPos pos) {
		tileEntities.remove(pos);
	}

	public void updateTileEntity(BlockPos pos, NBTTagCompound data) {
		if(tileEntities.containsKey(pos)) tileEntities.get(pos).readFromNBT(data);
		else System.out.println("Missing tile entity at: "+pos);
	}

	public Collection<TileEntity> getTileEntities() {
		return tileEntities.values();
	}

	public void reset() {
		Arrays.fill(blockBuffer, Blocks.AIR.getDefaultState());
		entities.clear();
		tileEntities.clear();
	}
}
