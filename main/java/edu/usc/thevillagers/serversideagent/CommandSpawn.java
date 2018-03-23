package edu.usc.thevillagers.serversideagent;

import java.util.UUID;

import com.mojang.authlib.GameProfile;

import edu.usc.thevillagers.serversideagent.agent.Agent;
import edu.usc.thevillagers.serversideagent.agent.AgentBrainExternal;
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
		return "/s <script>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 1)
			return;
		WorldServer world = server.worlds[0];
		BlockPos pos;
		if(args.length > 1)  
			pos = parseBlockPos(sender, args, 1, false);
		else
			pos = sender.getPosition();
		GameProfile profile = new GameProfile(new UUID(world.rand.nextLong(), world.rand.nextLong()), "Agent "+args[0]);
		Agent agent = new Agent(world, profile);
    	
    	agent.setPosition(pos.getX() + .5, pos.getY() + 1, pos.getZ() + .5);
		agent.connection = new NetHandlerPlayServer(world.getMinecraftServer(), new NetworkManager(EnumPacketDirection.SERVERBOUND), agent);
		
		server.getPlayerList().sendPacketToAllPlayers(new SPacketPlayerListItem(SPacketPlayerListItem.Action.ADD_PLAYER, new EntityPlayerMP[] {agent}));
		
		world.spawnEntity(agent);
		world.getPlayerChunkMap().addPlayer(agent);
		
		agent.setBrain(new AgentBrainExternal("python python/"+args[0]+".py"));
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}
