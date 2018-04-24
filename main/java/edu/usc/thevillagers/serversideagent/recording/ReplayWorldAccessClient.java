package edu.usc.thevillagers.serversideagent.recording;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.gui.ChunkBufferManager;
import net.minecraft.block.state.IBlockState;
import net.minecraft.client.Minecraft;
import net.minecraft.client.entity.EntityPlayerSP;
import net.minecraft.client.multiplayer.PlayerControllerMP;
import net.minecraft.client.multiplayer.WorldClient;
import net.minecraft.client.network.NetHandlerPlayClient;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetworkManager;
import net.minecraft.stats.RecipeBook;
import net.minecraft.stats.StatisticsManager;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.EnumDifficulty;
import net.minecraft.world.GameType;
import net.minecraft.world.World;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.WorldType;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

@SideOnly(value=Side.CLIENT)
public class ReplayWorldAccessClient extends ReplayWorldAccess {
	
	public EntityPlayerSP fakePlayer;
	public PlayerControllerMP fakePlayerController;
	public ChunkBufferManager chunkBufferManager;

	public ReplayWorldAccessClient(BlockPos from, BlockPos to) {
		super(from, to);
		chunkBufferManager = new ChunkBufferManager();
	}
	
	@Override
	public World createFakeWorld() {
		WorldSettings settings = new WorldSettings(0, GameType.NOT_SET, false, false, WorldType.FLAT);
		GameProfile profile = new GameProfile(UUID.randomUUID(), "dummy");
		Minecraft mc = Minecraft.getMinecraft();
		NetHandlerPlayClient nethandler = new NetHandlerPlayClient(mc, mc.currentScreen, new NetworkManager(EnumPacketDirection.CLIENTBOUND), profile);
		
		fakeWorld = new WorldClient(nethandler, settings, 0, EnumDifficulty.PEACEFUL, mc.mcProfiler) {
			
			@Override
			public IBlockState getBlockState(BlockPos pos) {
				return ReplayWorldAccessClient.this.getBlockState(pos);
			}
		};
		fakePlayer = new EntityPlayerSP(mc, fakeWorld, nethandler, new StatisticsManager(), new RecipeBook());
		fakePlayerController = new PlayerControllerMP(mc, nethandler);
		return fakeWorld;
	}

	
	@Override
	public void setBlockState(BlockPos pos, IBlockState state) {
		int index = index(pos);
		if(index >= 0 && !blockBuffer[index].equals(state)) {
			blockBuffer[index] = state;
			chunkBufferManager.requestUpdate(pos.getX() >> 4, pos.getY() >> 4, pos.getZ() >> 4);
		}
	}
}
