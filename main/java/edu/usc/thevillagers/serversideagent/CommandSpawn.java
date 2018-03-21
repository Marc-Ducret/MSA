package edu.usc.thevillagers.serversideagent;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.entity.player.EntityPlayerMP;
import net.minecraft.network.EnumPacketDirection;
import net.minecraft.network.NetHandlerPlayServer;
import net.minecraft.network.NetworkManager;
import net.minecraft.network.play.server.SPacketPlayerListItem;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.WorldServer;

public class CommandSpawn extends CommandBase {

	@Override
	public String getName() {
		return "s";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/s";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		WorldServer world = server.worlds[0];
		BlockPos pos = sender.getPosition();
		GameProfile profile = new GameProfile(new UUID(world.rand.nextLong(), world.rand.nextLong()), "Agent");
		EntityPlayerMP agent = new Agent(world, profile);
    	
    	agent.setPosition(pos.getX() + .5, pos.getY() + 1, pos.getZ() + .5);
		agent.connection = new NetHandlerPlayServer(world.getMinecraftServer(), new NetworkManager(EnumPacketDirection.SERVERBOUND), agent);
		
		server.getPlayerList().sendPacketToAllPlayers(new SPacketPlayerListItem(SPacketPlayerListItem.Action.ADD_PLAYER, new EntityPlayerMP[] {agent}));
		
		world.spawnEntity(agent);
		world.getPlayerChunkMap().addPlayer(agent);
	}
}
