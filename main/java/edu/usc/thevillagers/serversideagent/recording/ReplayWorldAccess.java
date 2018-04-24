package edu.usc.thevillagers.serversideagent.recording;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.gui.ChunkBufferManager;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.multiplayer.PlayerControllerMP;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.network.NetHandlerPlayClient;
import net.minecraft.entity.Entity;
import net.minecraft.init.Blocks;
import net.minecraft.nbt.NBTTagCompound;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetworkManager;
import net.minecraft.stats.RecipeBook;
import net.minecraft.stats.StatisticsManager;
import net.minecraft.tileentity.TileEntity;
import net.minecraft.util.EnumFacing;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.EnumDifficulty;
import net.minecraft.world.GameType;
import net.minecraft.world.IBlockAccess;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.WorldType;
import net.minecraft.world.biome.Biome;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

@SideOnly(value=Side.CLIENT)
public class ReplayWorldAccess implements IBlockAccess { //TODO refactor to be server-side friendly
	
	public final BlockPos from, to, diff;
	
	private final IBlockState[] blockBuffer;
	private final Map<Integer, Entity> entities;
	private final Map<BlockPos, TileEntity> tileEntities;
	
	public WorldClient fakeWorld;
	public EntityPlayerSP fakePlayer;
	public PlayerControllerMP fakePlayerController;
	public ChunkBufferManager chunkBufferManager;
	
	public ReplayWorldAccess(BlockPos from, BlockPos to) {
		this.from = from.toImmutable();
		this.to = to.toImmutable();
		
		diff = to.subtract(from).add(1, 1, 1);
		blockBuffer = new IBlockState[diff.getX() * diff.getY() * diff.getZ()];
		entities = new HashMap<>();
		tileEntities = new HashMap<>();
		
		WorldSettings settings = new WorldSettings(0, GameType.NOT_SET, false, false, WorldType.FLAT);
		GameProfile profile = new GameProfile(UUID.randomUUID(), "dummy");
		Minecraft mc = Minecraft.getMinecraft();
		NetHandlerPlayClient nethandler = new NetHandlerPlayClient(mc, mc.currentScreen, new NetworkManager(EnumPacketDirection.CLIENTBOUND), profile);
		fakeWorld = new WorldClient(nethandler, settings, 0, EnumDifficulty.PEACEFUL, mc.mcProfiler) {
			
			@Override
			public IBlockState getBlockState(BlockPos pos) {
				return ReplayWorldAccess.this.getBlockState(pos);
			}
		};
		fakePlayer = new EntityPlayerSP(mc, fakeWorld, nethandler, new StatisticsManager(), new RecipeBook());
		fakePlayerController = new PlayerControllerMP(mc, nethandler);
		chunkBufferManager = new ChunkBufferManager();
	}

	@Override
	public TileEntity getTileEntity(BlockPos pos) {
		return tileEntities.get(pos);
	}

	@Override
	public int getCombinedLight(BlockPos pos, int lightValue) {
		return 0xF << 4;
	}
	
	private int index(BlockPos pos) {
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
			chunkBufferManager.requestUpdate(pos.getX() >> 4, pos.getY() >> 4, pos.getZ() >> 4);
		}
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
		if(entities.containsKey(id)) entities.get(id).readFromNBT(data);
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
