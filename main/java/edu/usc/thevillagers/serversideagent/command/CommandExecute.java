package edu.usc.thevillagers.serversideagent.command;

import net.minecraft.command.CommandBase;
import net.minecraft.command.CommandException;
import net.minecraft.command.ICommandSender;
import net.minecraft.command.WrongUsageException;
import net.minecraft.server.MinecraftServer;

/**
 * Command that compiles a recording into a observation-action dataset
 */
public class CommandExecute extends CommandBase {
	
	@Override
	public String getName() {
		return "exec";
	}

	@Override
	public String getUsage(ICommandSender sender) {
		return "/exec <class> <static-method>";
	}

	@Override
	public void execute(MinecraftServer server, ICommandSender sender, String[] args) throws CommandException {
		if(args.length < 2) throw new WrongUsageException(getUsage(sender));
		try {
			Class.forName(args[0]).getMethod(args[1]).invoke(null);
		} catch (Exception e) {
			e.printStackTrace();
			throw new CommandException("An error occured: "+e.toString());
		}
	}
	
	@Override
	public int getRequiredPermissionLevel() {
		return 2;
	}
}
