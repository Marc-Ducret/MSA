package edu.usc.thevillagers.serversideagent;

import edu.usc.thevillagers.serversideagent.env.Environment;
import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.text.TextComponentString;
import net.minecraft.world.WorldServer;

public class CommandEnvironment extends CommandBase {
	
	private final ServerSideAgentMod mod;
	
	public CommandEnvironment(ServerSideAgentMod mod) {
		this.mod = mod;
	}

	@Override
	public String getName() {
		return "e";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/e <env> <agent>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 2) {
			sender.sendMessage(new TextComponentString("Incorrect usage ("+getUsage(sender)+")"));
			return;
		}
		WorldServer world = server.worlds[0];
		BlockPos pos;
		if(args.length > 2)  
			pos = parseBlockPos(sender, args, 1, false);
		else
			pos = sender.getPosition();
		try {
			Class<?> clazz = Class.forName("edu.usc.thevillagers.serversideagent.env.Environment"+args[0]);
			Environment env = (Environment) clazz.newInstance();
			env.setSpawnPoint(pos);
			String cmd = "python python/"+args[1]+".py";
			env.init(world, cmd);
			mod.addEnv(env);
		} catch (Exception e) {
			throw new CommandException("Env "+args[0]+" not found ("+e+")", e);
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}
